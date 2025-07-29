"""
Azure data source connectors for Data Processing service.
Handles connections to Azure SQL, Cosmos DB, Blob Storage, Event Hub, and other Azure services.
"""

import asyncio
import json
import uuid
import io
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import structlog
from azure.identity.aio import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient
from azure.cosmos.aio import CosmosClient
from azure.eventhub.aio import EventHubProducerClient, EventHubConsumerClient
from azure.servicebus.aio import ServiceBusClient
from azure.data.tables.aio import TableServiceClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from ....shared.config import get_settings
from ..models import DataSourceConfig, DataConnectorHealth, DataSourceType

settings = get_settings()
logger = structlog.get_logger(__name__)


class AzureConnectorService:
    """Service for managing Azure data source connections."""

    def __init__(self):
        self.settings = settings
        self.credential = None
        self.sql_engine = None
        self.cosmos_client = None
        self.blob_service_client = None
        self.eventhub_clients = {}
        self.servicebus_client = None
        self.table_service_client = None
        self._connection_pool = {}
        self._health_cache = {}
        self._last_health_check = {}

    async def _get_credential(self) -> DefaultAzureCredential:
        """Get Azure credential."""
        if self.credential is None:
            self.credential = DefaultAzureCredential()
        return self.credential

    async def _get_sql_engine(self):
        """Get SQL Server async engine."""
        if self.sql_engine is None:
            connection_string = self.settings.database.sql_connection_string
            self.sql_engine = create_async_engine(
                connection_string,
                pool_size=self.settings.database.sql_pool_size,
                max_overflow=self.settings.database.sql_max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600
            )
        return self.sql_engine

    async def _get_cosmos_client(self) -> CosmosClient:
        """Get Cosmos DB client."""
        if self.cosmos_client is None:
            self.cosmos_client = CosmosClient(
                url=self.settings.database.cosmos_endpoint,
                credential=self.settings.database.cosmos_key
            )
        return self.cosmos_client

    async def _get_blob_service_client(self) -> BlobServiceClient:
        """Get Blob Storage client."""
        if self.blob_service_client is None:
            credential = await self._get_credential()
            account_url = (
                f"https://{self.settings.azure.storage_account_name}.blob.core.windows.net"
            )
            self.blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=credential
            )
        return self.blob_service_client

    async def _get_eventhub_client(
        self,
        eventhub_name: str,
        connection_string: str
    ) -> EventHubProducerClient:
        """Get Event Hub client."""
        client_key = f"{eventhub_name}_{hash(connection_string)}"
        if client_key not in self.eventhub_clients:
            self.eventhub_clients[client_key] = EventHubProducerClient.from_connection_string(
                conn_str=connection_string,
                eventhub_name=eventhub_name
            )
        return self.eventhub_clients[client_key]

    async def _get_servicebus_client(self) -> ServiceBusClient:
        """Get Service Bus client."""
        if self.servicebus_client is None:
            self.servicebus_client = ServiceBusClient.from_connection_string(
                conn_str=self.settings.azure.service_bus_connection_string
            )
        return self.servicebus_client

    async def _get_table_service_client(self) -> TableServiceClient:
        """Get Table Storage client."""
        if self.table_service_client is None:
            credential = await self._get_credential()
            account_url = (
                f"https://{self.settings.azure.storage_account_name}.table.core.windows.net"
            )
            self.table_service_client = TableServiceClient(
                endpoint=account_url,
                credential=credential
            )
        return self.table_service_client

    async def connect_to_source(self, source_config: DataSourceConfig) -> Any:
        """Connect to data source based on configuration."""
        try:
            source_type = source_config.source_type

            if source_type == DataSourceType.AZURE_SQL:
                return await self._connect_to_azure_sql(source_config)
            elif source_type == DataSourceType.COSMOS_DB:
                return await self._connect_to_cosmos_db(source_config)
            elif source_type == DataSourceType.BLOB_STORAGE:
                return await self._connect_to_blob_storage(source_config)
            elif source_type == DataSourceType.EVENT_HUB:
                return await self._connect_to_event_hub(source_config)
            elif source_type == DataSourceType.SERVICE_BUS:
                return await self._connect_to_service_bus(source_config)
            elif source_type == DataSourceType.TABLE_STORAGE:
                return await self._connect_to_table_storage(source_config)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

        except Exception as e:
            logger.error(
                "connect_to_source_failed",
                source_type=source_type,
                error=str(e)
            )
            raise

    async def _connect_to_azure_sql(self, source_config: DataSourceConfig) -> AsyncSession:
        """Connect to Azure SQL Database."""
        try:
            engine = await self._get_sql_engine()
            SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            session = SessionLocal()

            # Test connection
            await session.execute(text("SELECT 1"))

            logger.info("azure_sql_connection_established")
            return session

        except Exception as e:
            logger.error("azure_sql_connection_failed", error=str(e))
            raise

    async def _connect_to_cosmos_db(self, source_config: DataSourceConfig) -> Any:
        """Connect to Cosmos DB."""
        try:
            client = await self._get_cosmos_client()
            database_name = source_config.database or self.settings.database.cosmos_database
            container_name = source_config.container

            database = client.get_database_client(database_name)
            container = database.get_container_client(container_name)

            # Test connection
            await container.read()

            logger.info(
                "cosmos_db_connection_established",
                database=database_name,
                container=container_name
            )
            return container

        except Exception as e:
            logger.error("cosmos_db_connection_failed", error=str(e))
            raise

    async def _connect_to_blob_storage(self, source_config: DataSourceConfig) -> Any:
        """Connect to Blob Storage."""
        try:
            client = await self._get_blob_service_client()
            container_name = source_config.container

            container_client = client.get_container_client(container_name)

            # Test connection
            await container_client.get_container_properties()

            logger.info(
                "blob_storage_connection_established",
                container=container_name
            )
            return container_client

        except Exception as e:
            logger.error("blob_storage_connection_failed", error=str(e))
            raise

    async def _connect_to_event_hub(self, source_config: DataSourceConfig) -> Any:
        """Connect to Event Hub."""
        try:
            connection_string = source_config.connection_string
            eventhub_name = source_config.additional_config.get("eventhub_name")

            if not connection_string or not eventhub_name:
                raise ValueError("Event Hub connection string and name are required")

            client = await self._get_eventhub_client(eventhub_name, connection_string)

            logger.info(
                "event_hub_connection_established",
                eventhub_name=eventhub_name
            )
            return client

        except Exception as e:
            logger.error("event_hub_connection_failed", error=str(e))
            raise

    async def _connect_to_service_bus(self, source_config: DataSourceConfig) -> Any:
        """Connect to Service Bus."""
        try:
            client = await self._get_servicebus_client()

            logger.info("service_bus_connection_established")
            return client

        except Exception as e:
            logger.error("service_bus_connection_failed", error=str(e))
            raise

    async def _connect_to_table_storage(self, source_config: DataSourceConfig) -> Any:
        """Connect to Table Storage."""
        try:
            client = await self._get_table_service_client()
            table_name = source_config.table

            table_client = client.get_table_client(table_name)

            # Test connection
            await table_client.get_entity("test", "test")

            logger.info(
                "table_storage_connection_established",
                table=table_name
            )
            return table_client

        except Exception as e:
            logger.error("table_storage_connection_failed", error=str(e))
            raise

    async def read_data(self, source_config: DataSourceConfig,
                       limit: Optional[int] = None,
                       offset: Optional[int] = None) -> pd.DataFrame:
        """Read data from source."""
        try:
            source_type = source_config.source_type

            if source_type == DataSourceType.AZURE_SQL:
                return await self._read_from_azure_sql(source_config, limit, offset)
            elif source_type == DataSourceType.COSMOS_DB:
                return await self._read_from_cosmos_db(source_config, limit, offset)
            elif source_type == DataSourceType.BLOB_STORAGE:
                return await self._read_from_blob_storage(source_config, limit, offset)
            elif source_type == DataSourceType.TABLE_STORAGE:
                return await self._read_from_table_storage(source_config, limit, offset)
            else:
                raise ValueError(f"Unsupported source type for reading: {source_type}")

        except Exception as e:
            logger.error(
                "read_data_failed",
                source_type=source_type,
                error=str(e)
            )
            raise

    async def _read_from_azure_sql(self, source_config: DataSourceConfig,
                                  limit: Optional[int] = None,
                                  offset: Optional[int] = None) -> pd.DataFrame:
        """Read data from Azure SQL Database."""
        try:
            session = await self._connect_to_azure_sql(source_config)

            # Build query
            query = source_config.query
            if not query:
                table_name = source_config.table
                query = f"SELECT * FROM {table_name}"

            # Add limit and offset
            if limit:
                query += f" OFFSET {offset or 0} ROWS FETCH NEXT {limit} ROWS ONLY"

            # Execute query
            result = await session.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()

            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=columns)

            await session.close()

            logger.info(
                "azure_sql_data_read",
                rows_count=len(df),
                columns_count=len(df.columns)
            )
            return df

        except Exception as e:
            logger.error("azure_sql_data_read_failed", error=str(e))
            raise

    async def _read_from_cosmos_db(self, source_config: DataSourceConfig,
                                  limit: Optional[int] = None,
                                  offset: Optional[int] = None) -> pd.DataFrame:
        """Read data from Cosmos DB."""
        try:
            container = await self._connect_to_cosmos_db(source_config)

            # Build query
            query = source_config.query or "SELECT * FROM c"

            # Add limit and offset
            if limit:
                query += f" OFFSET {offset or 0} LIMIT {limit}"

            # Execute query
            items = []
            async for item in container.query_items(query=query, enable_cross_partition_query=True):
                items.append(item)

            # Convert to DataFrame
            df = pd.DataFrame(items)

            logger.info(
                "cosmos_db_data_read",
                rows_count=len(df),
                columns_count=len(df.columns) if not df.empty else 0
            )
            return df

        except Exception as e:
            logger.error("cosmos_db_data_read_failed", error=str(e))
            raise

    async def _read_from_blob_storage(self, source_config: DataSourceConfig,
                                     limit: Optional[int] = None,
                                     offset: Optional[int] = None) -> pd.DataFrame:
        """Read data from Blob Storage."""
        try:
            container_client = await self._connect_to_blob_storage(source_config)

            # Get blob configuration
            blob_name = source_config.additional_config.get("blob_name")
            file_format = source_config.additional_config.get("file_format", "csv")

            if not blob_name:
                raise ValueError("Blob name is required for Blob Storage")

            # Download blob
            blob_client = container_client.get_blob_client(blob_name)
            blob_data = await blob_client.download_blob()
            content = await blob_data.readall()

            # Parse based on format
            if file_format.lower() == "csv":
                df = pd.read_csv(io.BytesIO(content))
            elif file_format.lower() == "json":
                df = pd.read_json(io.BytesIO(content))
            elif file_format.lower() == "parquet":
                df = pd.read_parquet(io.BytesIO(content))
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            # Apply limit and offset
            if offset:
                df = df.iloc[offset:]
            if limit:
                df = df.head(limit)

            logger.info(
                "blob_storage_data_read",
                blob_name=blob_name,
                rows_count=len(df),
                columns_count=len(df.columns)
            )
            return df

        except Exception as e:
            logger.error("blob_storage_data_read_failed", error=str(e))
            raise

    async def _read_from_table_storage(self, source_config: DataSourceConfig,
                                      limit: Optional[int] = None,
                                      offset: Optional[int] = None) -> pd.DataFrame:
        """Read data from Table Storage."""
        try:
            table_client = await self._connect_to_table_storage(source_config)

            # Build query
            query_filter = source_config.query

            # Query entities
            entities = []
            async for entity in table_client.query_entities(query_filter=query_filter):
                entities.append(entity)

            # Convert to DataFrame
            df = pd.DataFrame(entities)

            # Apply limit and offset
            if offset:
                df = df.iloc[offset:]
            if limit:
                df = df.head(limit)

            logger.info(
                "table_storage_data_read",
                table_name=source_config.table,
                rows_count=len(df),
                columns_count=len(df.columns) if not df.empty else 0
            )
            return df

        except Exception as e:
            logger.error("table_storage_data_read_failed", error=str(e))
            raise

    async def write_data(self, target_config: DataSourceConfig,
                        data: pd.DataFrame,
                        write_mode: str = "append") -> Dict[str, Any]:
        """Write data to target."""
        try:
            target_type = target_config.target_type

            if target_type == DataSourceType.AZURE_SQL:
                return await self._write_to_azure_sql(target_config, data, write_mode)
            elif target_type == DataSourceType.COSMOS_DB:
                return await self._write_to_cosmos_db(target_config, data, write_mode)
            elif target_type == DataSourceType.BLOB_STORAGE:
                return await self._write_to_blob_storage(target_config, data, write_mode)
            elif target_type == DataSourceType.TABLE_STORAGE:
                return await self._write_to_table_storage(target_config, data, write_mode)
            else:
                raise ValueError(f"Unsupported target type for writing: {target_type}")

        except Exception as e:
            logger.error(
                "write_data_failed",
                target_type=target_type,
                error=str(e)
            )
            raise

    async def _write_to_azure_sql(self, target_config: DataSourceConfig,
                                 data: pd.DataFrame,
                                 write_mode: str) -> Dict[str, Any]:
        """Write data to Azure SQL Database."""
        try:
            engine = await self._get_sql_engine()
            table_name = target_config.table

            if write_mode == "overwrite":
                if_exists = "replace"
            elif write_mode == "append":
                if_exists = "append"
            else:
                if_exists = "fail"

            # Write data
            await data.to_sql(
                name=table_name,
                con=engine,
                if_exists=if_exists,
                index=False,
                method="multi"
            )

            result = {
                "rows_written": len(data),
                "table_name": table_name,
                "write_mode": write_mode
            }

            logger.info(
                "azure_sql_data_written",
                **result
            )
            return result

        except Exception as e:
            logger.error("azure_sql_data_write_failed", error=str(e))
            raise

    async def _write_to_cosmos_db(self, target_config: DataSourceConfig,
                                 data: pd.DataFrame,
                                 write_mode: str) -> Dict[str, Any]:
        """Write data to Cosmos DB."""
        try:
            container = await self._connect_to_cosmos_db(target_config)

            # Convert DataFrame to list of dictionaries
            items = data.to_dict(orient="records")

            # Write items
            written_count = 0
            for item in items:
                # Add id if not present
                if "id" not in item:
                    item["id"] = str(uuid.uuid4())

                if write_mode == "overwrite":
                    await container.upsert_item(item)
                else:
                    await container.create_item(item)

                written_count += 1

            result = {
                "rows_written": written_count,
                "container_name": target_config.container,
                "write_mode": write_mode
            }

            logger.info(
                "cosmos_db_data_written",
                **result
            )
            return result

        except Exception as e:
            logger.error("cosmos_db_data_write_failed", error=str(e))
            raise

    async def _write_to_blob_storage(self, target_config: DataSourceConfig,
                                    data: pd.DataFrame,
                                    write_mode: str) -> Dict[str, Any]:
        """Write data to Blob Storage."""
        try:
            container_client = await self._connect_to_blob_storage(target_config)

            # Get blob configuration
            blob_name = target_config.additional_config.get("blob_name")
            file_format = target_config.additional_config.get("file_format", "csv")

            if not blob_name:
                blob_name = f"data_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{file_format}"

            # Convert DataFrame to bytes
            if file_format.lower() == "csv":
                content = data.to_csv(index=False).encode('utf-8')
            elif file_format.lower() == "json":
                content = data.to_json(orient="records").encode('utf-8')
            elif file_format.lower() == "parquet":
                import io
                buffer = io.BytesIO()
                data.to_parquet(buffer, index=False)
                content = buffer.getvalue()
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            # Upload blob
            blob_client = container_client.get_blob_client(blob_name)
            await blob_client.upload_blob(content, overwrite=(write_mode == "overwrite"))

            result = {
                "rows_written": len(data),
                "blob_name": blob_name,
                "file_size": len(content),
                "write_mode": write_mode
            }

            logger.info(
                "blob_storage_data_written",
                **result
            )
            return result

        except Exception as e:
            logger.error("blob_storage_data_write_failed", error=str(e))
            raise

    async def _write_to_table_storage(self, target_config: DataSourceConfig,
                                     data: pd.DataFrame,
                                     write_mode: str) -> Dict[str, Any]:
        """Write data to Table Storage."""
        try:
            table_client = await self._connect_to_table_storage(target_config)

            # Convert DataFrame to entities
            entities = data.to_dict(orient="records")

            # Write entities
            written_count = 0
            for entity in entities:
                # Add PartitionKey and RowKey if not present
                if "PartitionKey" not in entity:
                    entity["PartitionKey"] = "default"
                if "RowKey" not in entity:
                    entity["RowKey"] = str(uuid.uuid4())

                if write_mode == "overwrite":
                    await table_client.upsert_entity(entity)
                else:
                    await table_client.create_entity(entity)

                written_count += 1

            result = {
                "rows_written": written_count,
                "table_name": target_config.table,
                "write_mode": write_mode
            }

            logger.info(
                "table_storage_data_written",
                **result
            )
            return result

        except Exception as e:
            logger.error("table_storage_data_write_failed", error=str(e))
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all Azure connectors."""
        try:
            # Check if we have recent health data
            current_time = datetime.utcnow()
            if (self._last_health_check.get("timestamp") and
                current_time - self._last_health_check["timestamp"] < timedelta(minutes=5)):
                return self._health_cache

            health_status = {
                "status": "healthy",
                "connectors": [],
                "timestamp": current_time
            }

            # Check Azure SQL
            sql_health = await self._check_azure_sql_health()
            health_status["connectors"].append(sql_health)

            # Check Cosmos DB
            cosmos_health = await self._check_cosmos_db_health()
            health_status["connectors"].append(cosmos_health)

            # Check Blob Storage
            blob_health = await self._check_blob_storage_health()
            health_status["connectors"].append(blob_health)

            # Check Table Storage
            table_health = await self._check_table_storage_health()
            health_status["connectors"].append(table_health)

            # Determine overall status
            unhealthy_connectors = (
                [c for c in health_status["connectors"] if c["status"] != "healthy"]
            )
            if unhealthy_connectors:
                health_status["status"] = (
                    "degraded" if len(unhealthy_connectors) < len(health_status["connectors"]) else "unhealthy"
                )

            # Cache results
            self._health_cache = health_status
            self._last_health_check = {"timestamp": current_time}

            return health_status

        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "connectors": [],
                "timestamp": datetime.utcnow(),
                "error": str(e)
            }

    async def _check_azure_sql_health(self) -> DataConnectorHealth:
        """Check Azure SQL Database health."""
        try:
            start_time = datetime.utcnow()
            engine = await self._get_sql_engine()

            # Test query
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return DataConnectorHealth(
                connector_type="azure_sql",
                status="healthy",
                last_check=datetime.utcnow(),
                response_time_ms=int(response_time)
            )

        except Exception as e:
            return DataConnectorHealth(
                connector_type="azure_sql",
                status="unhealthy",
                last_check=datetime.utcnow(),
                error_message=str(e)
            )

    async def _check_cosmos_db_health(self) -> DataConnectorHealth:
        """Check Cosmos DB health."""
        try:
            start_time = datetime.utcnow()
            client = await self._get_cosmos_client()

            # Test operation
            await client.list_databases()

            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return DataConnectorHealth(
                connector_type="cosmos_db",
                status="healthy",
                last_check=datetime.utcnow(),
                response_time_ms=int(response_time)
            )

        except Exception as e:
            return DataConnectorHealth(
                connector_type="cosmos_db",
                status="unhealthy",
                last_check=datetime.utcnow(),
                error_message=str(e)
            )

    async def _check_blob_storage_health(self) -> DataConnectorHealth:
        """Check Blob Storage health."""
        try:
            start_time = datetime.utcnow()
            client = await self._get_blob_service_client()

            # Test operation
            async for container in client.list_containers():
                break

            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return DataConnectorHealth(
                connector_type="blob_storage",
                status="healthy",
                last_check=datetime.utcnow(),
                response_time_ms=int(response_time)
            )

        except Exception as e:
            return DataConnectorHealth(
                connector_type="blob_storage",
                status="unhealthy",
                last_check=datetime.utcnow(),
                error_message=str(e)
            )

    async def _check_table_storage_health(self) -> DataConnectorHealth:
        """Check Table Storage health."""
        try:
            start_time = datetime.utcnow()
            client = await self._get_table_service_client()

            # Test operation
            async for table in client.list_tables():
                break

            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return DataConnectorHealth(
                connector_type="table_storage",
                status="healthy",
                last_check=datetime.utcnow(),
                response_time_ms=int(response_time)
            )

        except Exception as e:
            return DataConnectorHealth(
                connector_type="table_storage",
                status="unhealthy",
                last_check=datetime.utcnow(),
                error_message=str(e)
            )

    async def close_connections(self) -> None:
        """Close all connections."""
        try:
            # Close SQL engine
            if self.sql_engine:
                await self.sql_engine.dispose()
                self.sql_engine = None

            # Close Cosmos client
            if self.cosmos_client:
                await self.cosmos_client.close()
                self.cosmos_client = None

            # Close Blob service client
            if self.blob_service_client:
                await self.blob_service_client.close()
                self.blob_service_client = None

            # Close Event Hub clients
            for client in self.eventhub_clients.values():
                await client.close()
            self.eventhub_clients.clear()

            # Close Service Bus client
            if self.servicebus_client:
                await self.servicebus_client.close()
                self.servicebus_client = None

            # Close Table service client
            if self.table_service_client:
                await self.table_service_client.close()
                self.table_service_client = None

            logger.info("azure_connectors_closed")

        except Exception as e:
            logger.error("close_connections_failed", error=str(e))
