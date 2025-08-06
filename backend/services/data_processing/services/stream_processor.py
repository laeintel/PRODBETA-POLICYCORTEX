"""
Stream processing service for real-time data processing.
Handles Event Hub, Service Bus, and other streaming data sources.
"""

import asyncio
import json
import uuid
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import structlog
from azure.eventhub import EventData
from azure.eventhub.aio import EventHubConsumerClient
from azure.eventhub.aio import EventHubProducerClient
from azure.servicebus import ServiceBusMessage
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus.aio import ServiceBusReceiver
from azure.servicebus.aio import ServiceBusSender
from sqlalchemy import text

from ....shared.config import get_settings
from ....shared.database import get_async_db
from ..models import DataSourceConfig
from ..models import DataTargetConfig
from ..models import PipelineStatus
from ..models import TransformationRule
from .data_transformer import DataTransformerService

settings = get_settings()
logger = structlog.get_logger(__name__)


class StreamProcessorService:
    """Service for managing stream processing operations."""

    def __init__(self):
        self.settings = settings
        self.data_transformer = DataTransformerService()
        self.active_processors = {}
        self.processor_tasks = {}
        self.processor_metrics = {}

    async def create_processor(self, source_config: DataSourceConfig,
                             processing_rules: List[TransformationRule],
                             output_config: DataTargetConfig,
                             window_size: Optional[int] = None,
                             window_type: str = "tumbling",
                             checkpoint_interval: int = 30,
                             parallelism: int = 1,
                             user_id: Optional[str] = None) -> str:
        """Create a new stream processor."""
        try:
            processor_id = str(uuid.uuid4())

            # Store processor configuration
            processor_config = {
                "processor_id": processor_id,
                "source_config": source_config.dict(),
                "processing_rules": [rule.dict() for rule in processing_rules],
                "output_config": output_config.dict(),
                "window_size": window_size,
                "window_type": window_type,
                "checkpoint_interval": checkpoint_interval,
                "parallelism": parallelism,
                "status": PipelineStatus.CREATED.value,
                "created_at": datetime.utcnow().isoformat(),
                "created_by": user_id,
                "last_updated": datetime.utcnow().isoformat()
            }

            # Save to database
            db = await get_async_db()
            await db.execute(
                text("""
                    INSERT INTO stream_processors (
                        processor_id, source_config, processing_rules, output_config,
                        window_size, window_type, checkpoint_interval, parallelism,
                        status, created_at, created_by, last_updated
                    ) VALUES (
                        :processor_id, :source_config, :processing_rules, :output_config,
                        :window_size, :window_type, :checkpoint_interval, :parallelism,
                        :status, :created_at, :created_by, :last_updated
                    )
                """),
                {
                    "processor_id": processor_id,
                    "source_config": json.dumps(processor_config["source_config"]),
                    "processing_rules": json.dumps(processor_config["processing_rules"]),
                    "output_config": json.dumps(processor_config["output_config"]),
                    "window_size": processor_config["window_size"],
                    "window_type": processor_config["window_type"],
                    "checkpoint_interval": processor_config["checkpoint_interval"],
                    "parallelism": processor_config["parallelism"],
                    "status": processor_config["status"],
                    "created_at": processor_config["created_at"],
                    "created_by": processor_config["created_by"],
                    "last_updated": processor_config["last_updated"]
                }
            )
            await db.commit()

            # Initialize processor metrics
            self.processor_metrics[processor_id] = {
                "messages_processed": 0,
                "messages_failed": 0,
                "bytes_processed": 0,
                "processing_rate": 0.0,
                "last_checkpoint": datetime.utcnow(),
                "uptime_seconds": 0
            }

            # Start the processor
            await self._start_processor(processor_id, processor_config)

            logger.info(
                "stream_processor_created",
                processor_id=processor_id,
                source_type=source_config.source_type,
                output_type=output_config.target_type
            )

            return processor_id

        except Exception as e:
            logger.error("stream_processor_creation_failed", error=str(e))
            raise

    async def _start_processor(self, processor_id: str, processor_config: Dict[str, Any]) -> None:
        """Start a stream processor."""
        try:
            source_config = DataSourceConfig(**processor_config["source_config"])

            # Create processor task based on source type
            if source_config.source_type == "event_hub":
                task = asyncio.create_task(
                    self._process_event_hub_stream(processor_id, processor_config)
                )
            elif source_config.source_type == "service_bus":
                task = asyncio.create_task(
                    self._process_service_bus_stream(processor_id, processor_config)
                )
            else:
                raise ValueError(f"Unsupported stream source type: {source_config.source_type}")

            self.processor_tasks[processor_id] = task
            self.active_processors[processor_id] = processor_config

            # Update status
            await self._update_processor_status(processor_id, PipelineStatus.RUNNING)

            logger.info("stream_processor_started", processor_id=processor_id)

        except Exception as e:
            logger.error("stream_processor_start_failed", error=str(e))
            await self._update_processor_status(processor_id, PipelineStatus.FAILED)
            raise

    async def _process_event_hub_stream(
        self,
        processor_id: str,
        processor_config: Dict[str,
        Any]
    ) -> None:
        """Process Event Hub stream."""
        try:
            source_config = DataSourceConfig(**processor_config["source_config"])
            output_config = DataTargetConfig(**processor_config["output_config"])
            processing_rules = (
                [TransformationRule(**rule) for rule in processor_config["processing_rules"]]
            )

            # Create Event Hub consumer
            connection_string = source_config.connection_string
            eventhub_name = source_config.additional_config.get("eventhub_name")
            consumer_group = source_config.additional_config.get("consumer_group", "$Default")

            async with EventHubConsumerClient.from_connection_string(
                conn_str=connection_string,
                consumer_group=consumer_group,
                eventhub_name=eventhub_name
            ) as client:

                # Create message handler
                async def on_event_batch(partition_context, event_batch):
                    try:
                        messages = []
                        for event in event_batch:
                            # Extract message data
                            message_data = {
                                "body": event.body_as_str(),
                                "properties": event.properties,
                                "system_properties": event.system_properties,
                                "partition_key": event.partition_key,
                                "offset": event.offset,
                                "sequence_number": event.sequence_number,
                                "enqueued_time": event.enqueued_time.isoformat() if event.enqueued_time else None
                            }

                            # Try to parse JSON body
                            try:
                                parsed_body = json.loads(event.body_as_str())
                                message_data["parsed_body"] = parsed_body
                            except json.JSONDecodeError:
                                pass

                            messages.append(message_data)

                        # Process batch
                        if messages:
                            await self._process_message_batch(
                                processor_id, messages, processing_rules, output_config
                            )

                        # Update checkpoint
                        await partition_context.update_checkpoint()

                    except Exception as e:
                        logger.error(
                            "event_hub_batch_processing_failed",
                            processor_id=processor_id,
                            error=str(e)
                        )
                        self.processor_metrics[processor_id]["messages_failed"] += len(event_batch)

                # Start receiving with batch processing
                await client.receive_batch(
                    on_event_batch=on_event_batch,
                    max_batch_size=100,
                    max_wait_time=30
                )

        except Exception as e:
            logger.error("event_hub_stream_processing_failed", error=str(e))
            await self._update_processor_status(processor_id, PipelineStatus.FAILED)
            raise

    async def _process_service_bus_stream(
        self,
        processor_id: str,
        processor_config: Dict[str,
        Any]
    ) -> None:
        """Process Service Bus stream."""
        try:
            source_config = DataSourceConfig(**processor_config["source_config"])
            output_config = DataTargetConfig(**processor_config["output_config"])
            processing_rules = (
                [TransformationRule(**rule) for rule in processor_config["processing_rules"]]
            )

            # Create Service Bus receiver
            connection_string = (
                source_config.connection_string or self.settings.azure.service_bus_connection_string
            )
            queue_name = source_config.additional_config.get("queue_name")
            topic_name = source_config.additional_config.get("topic_name")
            subscription_name = source_config.additional_config.get("subscription_name")

            async with ServiceBusClient.from_connection_string(connection_string) as client:

                # Create receiver based on configuration
                if queue_name:
                    receiver = client.get_queue_receiver(queue_name)
                elif topic_name and subscription_name:
                    receiver = client.get_subscription_receiver(topic_name, subscription_name)
                else:
                    raise ValueError("Either queue_name or (topic_name and
                        subscription_name) must be provided")

                async with receiver:
                    while processor_id in self.active_processors:
                        try:
                            # Receive messages
                            messages = await receiver.receive_messages(
                                max_message_count=10,
                                max_wait_time=30
                            )

                            if messages:
                                # Process messages
                                message_data = []
                                for message in messages:
                                    data = {
                                        "body": str(message),
                                        "properties": message.application_properties,
                                        "message_id": message.message_id,
                                        "correlation_id": message.correlation_id,
                                        "session_id": message.session_id,
                                        "delivery_count": message.delivery_count,
                                        "enqueued_time": message.enqueued_time_utc.isoformat() if message.enqueued_time_utc else None
                                    }

                                    # Try to parse JSON body
                                    try:
                                        parsed_body = json.loads(str(message))
                                        data["parsed_body"] = parsed_body
                                    except json.JSONDecodeError:
                                        pass

                                    message_data.append(data)

                                # Process batch
                                await self._process_message_batch(
                                    processor_id, message_data, processing_rules, output_config
                                )

                                # Complete messages
                                for message in messages:
                                    await receiver.complete_message(message)

                            # Small delay to prevent tight loop
                            await asyncio.sleep(1)

                        except Exception as e:
                            logger.error(
                                "service_bus_message_processing_failed",
                                processor_id=processor_id,
                                error=str(e)
                            )
                            if 'messages' in locals():
                                self.processor_metrics[processor_id]["messages_failed"] + = (
                                    len(messages)
                                )

        except Exception as e:
            logger.error("service_bus_stream_processing_failed", error=str(e))
            await self._update_processor_status(processor_id, PipelineStatus.FAILED)
            raise

    async def _process_message_batch(self, processor_id: str, messages: List[Dict[str, Any]],
                                   processing_rules: List[TransformationRule],
                                   output_config: DataTargetConfig) -> None:
        """Process a batch of messages."""
        try:
            start_time = datetime.utcnow()

            # Apply transformations
            if processing_rules:
                transformed_result = await self.data_transformer.transform_data(
                    data=messages,
                    transformation_rules=processing_rules,
                    output_format="json"
                )
                processed_messages = transformed_result["transformed_data"]
            else:
                processed_messages = messages

            # Send to output
            await self._send_to_output(processor_id, processed_messages, output_config)

            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            metrics = self.processor_metrics[processor_id]
            metrics["messages_processed"] += len(messages)
            metrics["bytes_processed"] += sum(len(json.dumps(msg)) for msg in messages)
            metrics["processing_rate"] = (
                len(messages) / processing_time if processing_time > 0 else 0
            )

            logger.debug(
                "message_batch_processed",
                processor_id=processor_id,
                message_count=len(messages),
                processing_time=processing_time
            )

        except Exception as e:
            logger.error("message_batch_processing_failed", error=str(e))
            self.processor_metrics[processor_id]["messages_failed"] += len(messages)
            raise

    async def _send_to_output(self, processor_id: str, messages: List[Dict[str, Any]],
                            output_config: DataTargetConfig) -> None:
        """Send processed messages to output destination."""
        try:
            output_type = output_config.target_type

            if output_type == "event_hub":
                await self._send_to_event_hub(messages, output_config)
            elif output_type == "service_bus":
                await self._send_to_service_bus(messages, output_config)
            elif output_type == "blob_storage":
                await self._send_to_blob_storage(processor_id, messages, output_config)
            elif output_type == "cosmos_db":
                await self._send_to_cosmos_db(messages, output_config)
            else:
                raise ValueError(f"Unsupported output type: {output_type}")

        except Exception as e:
            logger.error("send_to_output_failed", error=str(e))
            raise

    async def _send_to_event_hub(
        self,
        messages: List[Dict[str,
        Any]],
        output_config: DataTargetConfig
    ) -> None:
        """Send messages to Event Hub."""
        try:
            connection_string = output_config.connection_string
            eventhub_name = output_config.additional_config.get("eventhub_name")

            async with EventHubProducerClient.from_connection_string(
                conn_str=connection_string,
                eventhub_name=eventhub_name
            ) as producer:

                # Create event batch
                event_batch = await producer.create_batch()

                for message in messages:
                    event_data = EventData(json.dumps(message))

                    # Add partition key if specified
                    partition_key = message.get("partition_key")
                    if partition_key:
                        event_data.partition_key = partition_key

                    try:
                        event_batch.add(event_data)
                    except ValueError:
                        # Batch is full, send it and create a new one
                        await producer.send_batch(event_batch)
                        event_batch = await producer.create_batch()
                        event_batch.add(event_data)

                # Send remaining events
                if len(event_batch) > 0:
                    await producer.send_batch(event_batch)

        except Exception as e:
            logger.error("send_to_event_hub_failed", error=str(e))
            raise

    async def _send_to_service_bus(
        self,
        messages: List[Dict[str,
        Any]],
        output_config: DataTargetConfig
    ) -> None:
        """Send messages to Service Bus."""
        try:
            connection_string = (
                output_config.connection_string or self.settings.azure.service_bus_connection_string
            )
            queue_name = output_config.additional_config.get("queue_name")
            topic_name = output_config.additional_config.get("topic_name")

            async with ServiceBusClient.from_connection_string(connection_string) as client:

                # Create sender based on configuration
                if queue_name:
                    sender = client.get_queue_sender(queue_name)
                elif topic_name:
                    sender = client.get_topic_sender(topic_name)
                else:
                    raise ValueError("Either queue_name or topic_name must be provided")

                async with sender:
                    # Send messages
                    service_bus_messages = []
                    for message in messages:
                        sb_message = ServiceBusMessage(json.dumps(message))

                        # Add message properties
                        if "message_id" in message:
                            sb_message.message_id = message["message_id"]
                        if "correlation_id" in message:
                            sb_message.correlation_id = message["correlation_id"]
                        if "session_id" in message:
                            sb_message.session_id = message["session_id"]

                        service_bus_messages.append(sb_message)

                    await sender.send_messages(service_bus_messages)

        except Exception as e:
            logger.error("send_to_service_bus_failed", error=str(e))
            raise

    async def _send_to_blob_storage(self, processor_id: str, messages: List[Dict[str, Any]],
                                  output_config: DataTargetConfig) -> None:
        """Send messages to Blob Storage."""
        try:
            # Use Azure connector to write to blob storage
            import pandas as pd

            from .azure_connectors import AzureConnectorService

            azure_connector = AzureConnectorService()

            # Convert messages to DataFrame
            df = pd.DataFrame(messages)

            # Generate blob name with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            blob_name = f"stream_data_{processor_id}_{timestamp}.json"

            # Update output config with blob name
            output_config.additional_config = output_config.additional_config or {}
            output_config.additional_config["blob_name"] = blob_name
            output_config.additional_config["file_format"] = "json"

            # Write to blob storage
            await azure_connector.write_data(output_config, df, "append")

        except Exception as e:
            logger.error("send_to_blob_storage_failed", error=str(e))
            raise

    async def _send_to_cosmos_db(
        self,
        messages: List[Dict[str,
        Any]],
        output_config: DataTargetConfig
    ) -> None:
        """Send messages to Cosmos DB."""
        try:
            # Use Azure connector to write to Cosmos DB
            import pandas as pd

            from .azure_connectors import AzureConnectorService

            azure_connector = AzureConnectorService()

            # Convert messages to DataFrame
            df = pd.DataFrame(messages)

            # Write to Cosmos DB
            await azure_connector.write_data(output_config, df, "append")

        except Exception as e:
            logger.error("send_to_cosmos_db_failed", error=str(e))
            raise

    async def _update_processor_status(self, processor_id: str, status: PipelineStatus) -> None:
        """Update processor status."""
        try:
            db = await get_async_db()
            await db.execute(
                text("""
                    UPDATE stream_processors
                    SET status = :status, last_updated = :last_updated
                    WHERE processor_id = :processor_id
                """),
                {
                    "processor_id": processor_id,
                    "status": status.value,
                    "last_updated": datetime.utcnow().isoformat()
                }
            )
            await db.commit()

        except Exception as e:
            logger.error("update_processor_status_failed", error=str(e))

    async def stop_processor(self, processor_id: str) -> None:
        """Stop a stream processor."""
        try:
            # Cancel processor task
            if processor_id in self.processor_tasks:
                task = self.processor_tasks[processor_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self.processor_tasks[processor_id]

            # Remove from active processors
            if processor_id in self.active_processors:
                del self.active_processors[processor_id]

            # Update status
            await self._update_processor_status(processor_id, PipelineStatus.CANCELLED)

            logger.info("stream_processor_stopped", processor_id=processor_id)

        except Exception as e:
            logger.error("stream_processor_stop_failed", error=str(e))
            raise

    async def get_processor_info(self, processor_id: str) -> Dict[str, Any]:
        """Get processor information."""
        try:
            # Get processor configuration
            db = await get_async_db()
            result = await db.execute(
                text("SELECT * FROM stream_processors WHERE processor_id = :processor_id"),
                {"processor_id": processor_id}
            )
            row = result.fetchone()

            if not row:
                raise ValueError(f"Processor {processor_id} not found")

            # Get metrics
            metrics = self.processor_metrics.get(processor_id, {})

            return {
                "processor_id": processor_id,
                "status": row.status,
                "created_at": row.created_at,
                "last_updated": row.last_updated,
                "metrics": metrics,
                "message": f"Processor {processor_id} information retrieved successfully"
            }

        except Exception as e:
            logger.error("get_processor_info_failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close all stream processors."""
        try:
            # Stop all active processors
            for processor_id in list(self.active_processors.keys()):
                await self.stop_processor(processor_id)

            logger.info("stream_processor_service_closed")

        except Exception as e:
            logger.error("stream_processor_service_close_failed", error=str(e))
