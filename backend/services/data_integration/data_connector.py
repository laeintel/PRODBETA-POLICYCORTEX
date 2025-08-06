"""
Data Connector Module
Handles connections to various data sources
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import aiohttp
import asyncpg
import motor.motor_asyncio
import pandas as pd
import structlog
from azure.cosmos.aio import CosmosClient
from azure.eventhub.aio import EventHubConsumerClient
from azure.eventhub.aio import EventHubProducerClient
from azure.servicebus.aio import ServiceBusClient
from azure.storage.blob.aio import BlobServiceClient

logger = structlog.get_logger(__name__)


class DataSourceType(str, Enum):
    """Supported data source types"""

    AZURE_BLOB = "azure_blob"
    AZURE_COSMOS = "azure_cosmos"
    AZURE_SQL = "azure_sql"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REST_API = "rest_api"
    EVENT_HUB = "event_hub"
    SERVICE_BUS = "service_bus"
    KAFKA = "kafka"
    S3 = "s3"
    ELASTICSEARCH = "elasticsearch"


class DataConnector:
    """
    Universal data connector for various data sources
    """

    def __init__(self):
        self.connections = {}
        self.connection_pools = {}

    async def connect(
        self, source_name: str, source_type: DataSourceType, connection_config: Dict[str, Any]
    ) -> bool:
        """
        Establish connection to a data source

        Args:
            source_name: Unique identifier for the connection
            source_type: Type of data source
            connection_config: Configuration parameters

        Returns:
            Success status
        """

        try:
            if source_type == DataSourceType.AZURE_BLOB:
                connection = await self._connect_azure_blob(connection_config)

            elif source_type == DataSourceType.AZURE_COSMOS:
                connection = await self._connect_azure_cosmos(connection_config)

            elif source_type == DataSourceType.AZURE_SQL:
                connection = await self._connect_azure_sql(connection_config)

            elif source_type == DataSourceType.POSTGRESQL:
                connection = await self._connect_postgresql(connection_config)

            elif source_type == DataSourceType.MONGODB:
                connection = await self._connect_mongodb(connection_config)

            elif source_type == DataSourceType.REST_API:
                connection = await self._connect_rest_api(connection_config)

            elif source_type == DataSourceType.EVENT_HUB:
                connection = await self._connect_event_hub(connection_config)

            elif source_type == DataSourceType.SERVICE_BUS:
                connection = await self._connect_service_bus(connection_config)

            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            self.connections[source_name] = {
                "type": source_type,
                "connection": connection,
                "config": connection_config,
                "connected_at": datetime.utcnow(),
            }

            logger.info(f"Connected to {source_name} ({source_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {source_name}: {e}")
            return False

    async def disconnect(self, source_name: str) -> bool:
        """Disconnect from a data source"""

        if source_name not in self.connections:
            return False

        try:
            conn_info = self.connections[source_name]
            source_type = conn_info["type"]
            connection = conn_info["connection"]

            # Close connection based on type
            if source_type == DataSourceType.POSTGRESQL:
                await connection.close()
            elif source_type == DataSourceType.MONGODB:
                connection.close()
            elif source_type in [DataSourceType.EVENT_HUB, DataSourceType.SERVICE_BUS]:
                await connection.close()
            elif source_type == DataSourceType.REST_API:
                await connection.close()

            del self.connections[source_name]
            logger.info(f"Disconnected from {source_name}")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from {source_name}: {e}")
            return False

    async def read_data(
        self,
        source_name: str,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Read data from a connected source

        Args:
            source_name: Name of the data source
            query: Query string (SQL, MongoDB query, etc.)
            filters: Additional filters
            limit: Maximum records to retrieve

        Returns:
            DataFrame with retrieved data
        """

        if source_name not in self.connections:
            raise ValueError(f"Source {source_name} not connected")

        conn_info = self.connections[source_name]
        source_type = conn_info["type"]
        connection = conn_info["connection"]

        if source_type == DataSourceType.POSTGRESQL:
            return await self._read_postgresql(connection, query, limit)

        elif source_type == DataSourceType.AZURE_COSMOS:
            return await self._read_cosmos(connection, conn_info["config"], query, filters, limit)

        elif source_type == DataSourceType.MONGODB:
            return await self._read_mongodb(connection, conn_info["config"], query, filters, limit)

        elif source_type == DataSourceType.REST_API:
            return await self._read_rest_api(connection, conn_info["config"], filters)

        elif source_type == DataSourceType.AZURE_BLOB:
            return await self._read_blob(connection, conn_info["config"], filters)

        else:
            raise ValueError(f"Read not implemented for {source_type}")

    async def write_data(
        self,
        source_name: str,
        data: pd.DataFrame,
        table_name: Optional[str] = None,
        mode: str = "append",
    ) -> bool:
        """
        Write data to a connected source

        Args:
            source_name: Name of the data source
            data: Data to write
            table_name: Target table/collection name
            mode: Write mode (append, replace, upsert)

        Returns:
            Success status
        """

        if source_name not in self.connections:
            raise ValueError(f"Source {source_name} not connected")

        conn_info = self.connections[source_name]
        source_type = conn_info["type"]
        connection = conn_info["connection"]

        try:
            if source_type == DataSourceType.POSTGRESQL:
                return await self._write_postgresql(connection, data, table_name, mode)

            elif source_type == DataSourceType.AZURE_COSMOS:
                return await self._write_cosmos(connection, conn_info["config"], data, mode)

            elif source_type == DataSourceType.MONGODB:
                return await self._write_mongodb(
                    connection, conn_info["config"], data, table_name, mode
                )

            elif source_type == DataSourceType.AZURE_BLOB:
                return await self._write_blob(connection, conn_info["config"], data)

            else:
                raise ValueError(f"Write not implemented for {source_type}")

        except Exception as e:
            logger.error(f"Failed to write data to {source_name}: {e}")
            return False

    # Connection methods for different data sources

    async def _connect_azure_blob(self, config: Dict[str, Any]) -> BlobServiceClient:
        """Connect to Azure Blob Storage"""
        return BlobServiceClient(
            account_url=f"https://{config['account_name']}.blob.core.windows.net",
            credential=config.get("credential"),
        )

    async def _connect_azure_cosmos(self, config: Dict[str, Any]) -> CosmosClient:
        """Connect to Azure Cosmos DB"""
        return CosmosClient(url=config["endpoint"], credential=config["key"])

    async def _connect_azure_sql(self, config: Dict[str, Any]):
        """Connect to Azure SQL Database"""
        import pyodbc

        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={config['server']}.database.windows.net;"
            f"DATABASE={config['database']};"
            f"UID={config['username']};"
            f"PWD={config['password']}"
        )

        return pyodbc.connect(connection_string)

    async def _connect_postgresql(self, config: Dict[str, Any]) -> asyncpg.Connection:
        """Connect to PostgreSQL"""
        return await asyncpg.connect(
            host=config["host"],
            port=config.get("port", 5432),
            user=config["user"],
            password=config["password"],
            database=config["database"],
        )

    async def _connect_mongodb(self, config: Dict[str, Any]):
        """Connect to MongoDB"""
        client = motor.motor_asyncio.AsyncIOMotorClient(config["connection_string"])
        return client[config["database"]]

    async def _connect_rest_api(self, config: Dict[str, Any]) -> aiohttp.ClientSession:
        """Connect to REST API"""
        headers = config.get("headers", {})
        auth = None

        if "api_key" in config:
            headers["Authorization"] = f"Bearer {config['api_key']}"

        return aiohttp.ClientSession(base_url=config["base_url"], headers=headers)

    async def _connect_event_hub(self, config: Dict[str, Any]):
        """Connect to Azure Event Hub"""
        if config.get("mode") == "producer":
            return EventHubProducerClient.from_connection_string(
                config["connection_string"], eventhub_name=config["event_hub_name"]
            )
        else:
            return EventHubConsumerClient.from_connection_string(
                config["connection_string"],
                consumer_group=config.get("consumer_group", "$Default"),
                eventhub_name=config["event_hub_name"],
            )

    async def _connect_service_bus(self, config: Dict[str, Any]) -> ServiceBusClient:
        """Connect to Azure Service Bus"""
        return ServiceBusClient.from_connection_string(config["connection_string"])

    # Read methods for different data sources

    async def _read_postgresql(
        self, connection: asyncpg.Connection, query: str, limit: Optional[int]
    ) -> pd.DataFrame:
        """Read from PostgreSQL"""
        if limit:
            query = f"{query} LIMIT {limit}"

        rows = await connection.fetch(query)
        return pd.DataFrame([dict(row) for row in rows])

    async def _read_cosmos(
        self,
        client: CosmosClient,
        config: Dict[str, Any],
        query: Optional[str],
        filters: Optional[Dict[str, Any]],
        limit: Optional[int],
    ) -> pd.DataFrame:
        """Read from Cosmos DB"""
        database = client.get_database_client(config["database"])
        container = database.get_container_client(config["container"])

        if query:
            items = container.query_items(query=query, enable_cross_partition_query=True)
        else:
            items = container.read_all_items()

        data = []
        async for item in items:
            data.append(item)
            if limit and len(data) >= limit:
                break

        return pd.DataFrame(data)

    async def _read_mongodb(
        self,
        db,
        config: Dict[str, Any],
        query: Optional[Dict],
        filters: Optional[Dict[str, Any]],
        limit: Optional[int],
    ) -> pd.DataFrame:
        """Read from MongoDB"""
        collection = db[config["collection"]]

        cursor = collection.find(query or {})

        if limit:
            cursor = cursor.limit(limit)

        data = []
        async for document in cursor:
            document["_id"] = str(document["_id"])  # Convert ObjectId to string
            data.append(document)

        return pd.DataFrame(data)

    async def _read_rest_api(
        self,
        session: aiohttp.ClientSession,
        config: Dict[str, Any],
        filters: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Read from REST API"""
        endpoint = config.get("endpoint", "")
        params = filters or {}

        async with session.get(endpoint, params=params) as response:
            data = await response.json()

            # Handle pagination if configured
            if config.get("pagination"):
                all_data = data.get(config["data_field"], data)

                while config["pagination"].get("next_field") in data:
                    next_url = data[config["pagination"]["next_field"]]
                    if not next_url:
                        break

                    async with session.get(next_url) as response:
                        data = await response.json()
                        all_data.extend(data.get(config["data_field"], data))

                return pd.DataFrame(all_data)
            else:
                return pd.DataFrame(data.get(config.get("data_field"), data))

    async def _read_blob(
        self, client: BlobServiceClient, config: Dict[str, Any], filters: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Read from Azure Blob Storage"""
        container_client = client.get_container_client(config["container"])
        blob_name = filters.get("blob_name") if filters else config.get("blob_name")

        blob_client = container_client.get_blob_client(blob_name)
        download_stream = await blob_client.download_blob()
        content = await download_stream.readall()

        # Parse based on file type
        if blob_name.endswith(".csv"):
            import io

            return pd.read_csv(io.BytesIO(content))
        elif blob_name.endswith(".json"):
            import json

            return pd.DataFrame(json.loads(content))
        elif blob_name.endswith(".parquet"):
            import io

            return pd.read_parquet(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported file type for {blob_name}")

    # Write methods for different data sources

    async def _write_postgresql(
        self, connection: asyncpg.Connection, data: pd.DataFrame, table_name: str, mode: str
    ) -> bool:
        """Write to PostgreSQL"""
        if mode == "replace":
            await connection.execute(f"TRUNCATE TABLE {table_name}")

        # Convert DataFrame to records
        records = data.to_dict("records")

        # Build insert query
        columns = list(data.columns)
        placeholders = [f"${i+1}" for i in range(len(columns))]

        if mode == "upsert":
            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT DO UPDATE SET
                {', '.join([f"{col} = EXCLUDED.{col}" for col in columns])}
            """
        else:
            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """

        # Execute batch insert
        await connection.executemany(query, [tuple(r.values()) for r in records])

        return True

    async def _write_cosmos(
        self, client: CosmosClient, config: Dict[str, Any], data: pd.DataFrame, mode: str
    ) -> bool:
        """Write to Cosmos DB"""
        database = client.get_database_client(config["database"])
        container = database.get_container_client(config["container"])

        records = data.to_dict("records")

        for record in records:
            if mode == "upsert":
                await container.upsert_item(record)
            else:
                await container.create_item(record)

        return True

    async def _write_mongodb(
        self, db, config: Dict[str, Any], data: pd.DataFrame, collection_name: str, mode: str
    ) -> bool:
        """Write to MongoDB"""
        collection = db[collection_name or config["collection"]]

        if mode == "replace":
            await collection.delete_many({})

        records = data.to_dict("records")

        if mode == "upsert":
            for record in records:
                await collection.replace_one(
                    {"_id": record.get("_id", record.get("id"))}, record, upsert=True
                )
        else:
            await collection.insert_many(records)

        return True

    async def _write_blob(
        self, client: BlobServiceClient, config: Dict[str, Any], data: pd.DataFrame
    ) -> bool:
        """Write to Azure Blob Storage"""
        container_client = client.get_container_client(config["container"])

        # Generate blob name with timestamp
        blob_name = (
            f"{config.get('prefix', 'data')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        )

        blob_client = container_client.get_blob_client(blob_name)

        # Convert DataFrame to bytes
        import io

        buffer = io.BytesIO()
        data.to_parquet(buffer)

        # Upload
        await blob_client.upload_blob(buffer.getvalue(), overwrite=True)

        return True

    async def stream_data(self, source_name: str, callback, batch_size: int = 100):
        """
        Stream data from a source

        Args:
            source_name: Name of the data source
            callback: Async function to process each batch
            batch_size: Size of each batch
        """

        if source_name not in self.connections:
            raise ValueError(f"Source {source_name} not connected")

        conn_info = self.connections[source_name]
        source_type = conn_info["type"]
        connection = conn_info["connection"]

        if source_type == DataSourceType.EVENT_HUB:
            await self._stream_event_hub(connection, callback, batch_size)
        elif source_type == DataSourceType.SERVICE_BUS:
            await self._stream_service_bus(connection, conn_info["config"], callback, batch_size)
        else:
            raise ValueError(f"Streaming not supported for {source_type}")

    async def _stream_event_hub(self, client: EventHubConsumerClient, callback, batch_size: int):
        """Stream from Event Hub"""

        async def on_event_batch(partition_context, events):
            batch = []
            for event in events:
                batch.append(event.body_as_json())

                if len(batch) >= batch_size:
                    await callback(pd.DataFrame(batch))
                    batch = []

            if batch:
                await callback(pd.DataFrame(batch))

            await partition_context.update_checkpoint()

        async with client:
            await client.receive_batch(on_event_batch=on_event_batch, max_batch_size=batch_size)

    async def _stream_service_bus(
        self, client: ServiceBusClient, config: Dict[str, Any], callback, batch_size: int
    ):
        """Stream from Service Bus"""

        async with client:
            receiver = client.get_queue_receiver(queue_name=config.get("queue_name"))

            async with receiver:
                batch = []

                async for message in receiver:
                    batch.append(json.loads(str(message)))

                    if len(batch) >= batch_size:
                        await callback(pd.DataFrame(batch))
                        batch = []
                        await receiver.complete_message(message)

                if batch:
                    await callback(pd.DataFrame(batch))
