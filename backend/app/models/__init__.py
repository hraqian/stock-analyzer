# backend/app/models/__init__.py
# Import all ORM models so Base.metadata.create_all() creates their tables.
from app.models.user import User  # noqa: F401
from app.models.strategy import Strategy  # noqa: F401
