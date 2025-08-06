"""
Shared database module for PolicyCortex microservices.
Provides SQLAlchemy configuration, base models, and database utilities.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator
from typing import Optional

from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .config import get_settings

settings = get_settings()

# Base class for all database models
Base = declarative_base()


class BaseModel(Base):
    """Base model with common fields for all entities."""

    __abstract__ = True

    id = Column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_by = Column(String(255), nullable=True)
    updated_by = Column(String(255), nullable=True)

    def to_dict(self) -> dict:
        """Convert model instance to dictionary."""
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__}(id={self.id})>"


class AuditLog(BaseModel):
    """Audit log model for tracking changes and actions."""

    __tablename__ = "audit_logs"

    entity_type = Column(String(100), nullable=False)
    entity_id = Column(String(255), nullable=False)
    action = Column(String(50), nullable=False)  # CREATE, UPDATE, DELETE, VIEW
    old_values = Column(Text, nullable=True)  # JSON string of old values
    new_values = Column(Text, nullable=True)  # JSON string of new values
    user_id = Column(String(255), nullable=True)
    session_id = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    details = Column(Text, nullable=True)


class UserSession(BaseModel):
    """User session model for tracking active sessions."""

    __tablename__ = "user_sessions"

    user_id = Column(String(255), nullable=False)
    session_token = Column(String(500), nullable=False, unique=True)
    refresh_token = Column(String(500), nullable=True)
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    is_revoked = Column(Boolean, default=False, nullable=False)


class ServiceHealth(BaseModel):
    """Service health tracking model."""

    __tablename__ = "service_health"

    service_name = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)  # HEALTHY, DEGRADED, UNHEALTHY
    last_check = Column(DateTime, default=datetime.utcnow, nullable=False)
    response_time_ms = Column(Integer, nullable=True)
    error_count = Column(Integer, default=0, nullable=False)
    details = Column(Text, nullable=True)


# Database Engine Configuration
def create_database_engine(connection_string: str, is_async: bool = False):
    """Create database engine with appropriate configuration."""

    if is_async:
        engine = create_async_engine(
            connection_string,
            echo=settings.debug,
            pool_size=settings.database.sql_pool_size,
            max_overflow=settings.database.sql_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
    else:
        engine = create_engine(
            connection_string,
            echo=settings.debug,
            pool_size=settings.database.sql_pool_size,
            max_overflow=settings.database.sql_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    return engine


# Create engines
async_engine = create_database_engine(
    settings.database.sql_connection_string.replace("mssql+pyodbc", "mssql+aioodbc"), is_async=True
)

sync_engine = create_database_engine(settings.database.sql_connection_string, is_async=False)

# Session factories
AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


class DatabaseManager:
    """Database manager for handling connections and operations."""

    def __init__(self):
        self.async_engine = async_engine
        self.sync_engine = sync_engine

    async def create_tables(self):
        """Create all database tables."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self):
        """Drop all database tables."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def health_check(self) -> dict:
        """Perform database health check."""
        try:
            async with AsyncSessionLocal() as session:
                await session.execute("SELECT 1")
                return {"status": "healthy", "message": "Database connection successful"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Database connection failed: {str(e)}"}


# Dependency for getting async database session
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Dependency for getting sync database session
def get_sync_db() -> Session:
    """Dependency for getting sync database session."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# Context manager for database transactions
@asynccontextmanager
async def async_db_transaction() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for async database transactions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Database utilities
class DatabaseUtils:
    """Utility functions for database operations."""

    @staticmethod
    async def log_audit_event(
        session: AsyncSession,
        entity_type: str,
        entity_id: str,
        action: str,
        user_id: Optional[str] = None,
        old_values: Optional[dict] = None,
        new_values: Optional[dict] = None,
        details: Optional[str] = None,
    ):
        """Log an audit event."""
        import json

        audit_log = AuditLog(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            user_id=user_id,
            old_values=json.dumps(old_values) if old_values else None,
            new_values=json.dumps(new_values) if new_values else None,
            details=details,
        )

        session.add(audit_log)
        await session.flush()

    @staticmethod
    async def update_service_health(
        session: AsyncSession,
        service_name: str,
        status: str,
        response_time_ms: Optional[int] = None,
        error_count: int = 0,
        details: Optional[str] = None,
    ):
        """Update service health status."""
        from sqlalchemy import select
        from sqlalchemy import update

        # Check if record exists
        stmt = select(ServiceHealth).where(ServiceHealth.service_name == service_name)
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing record
            update_stmt = (
                update(ServiceHealth)
                .where(ServiceHealth.service_name == service_name)
                .values(
                    status=status,
                    last_check=datetime.utcnow(),
                    response_time_ms=response_time_ms,
                    error_count=error_count,
                    details=details,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.execute(update_stmt)
        else:
            # Create new record
            health_record = ServiceHealth(
                service_name=service_name,
                status=status,
                response_time_ms=response_time_ms,
                error_count=error_count,
                details=details,
            )
            session.add(health_record)

        await session.flush()


# Global database manager instance
db_manager = DatabaseManager()


# Cosmos DB Configuration (for NoSQL data)
class CosmosDBManager:
    """Manager for Cosmos DB operations."""

    def __init__(self):
        self.endpoint = settings.database.cosmos_endpoint
        self.key = settings.database.cosmos_key
        self.database_name = settings.database.cosmos_database
        self._client = None
        self._database = None

    async def get_client(self):
        """Get Cosmos DB client."""
        if self._client is None:
            from azure.cosmos.aio import CosmosClient

            self._client = CosmosClient(self.endpoint, self.key)
        return self._client

    async def get_database(self):
        """Get Cosmos DB database."""
        if self._database is None:
            client = await self.get_client()
            self._database = client.get_database_client(self.database_name)
        return self._database

    async def get_container(self, container_name: str):
        """Get Cosmos DB container."""
        database = await self.get_database()
        return database.get_container_client(container_name)

    async def health_check(self) -> dict:
        """Perform Cosmos DB health check."""
        try:
            client = await self.get_client()
            await client.get_database_account()
            return {"status": "healthy", "message": "Cosmos DB connection successful"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Cosmos DB connection failed: {str(e)}"}


# Global Cosmos DB manager instance
cosmos_manager = CosmosDBManager()
