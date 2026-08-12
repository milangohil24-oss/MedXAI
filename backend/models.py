
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
import datetime

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )

    analyses = relationship(
        "Analysis",
        back_populates="user"
    )

    reports = relationship(
        "Report",
        back_populates="user"
    )


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(
        String,
        ForeignKey("users.id")
    )

    filename = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    confidence_percentage = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)

    gradcam_url = Column(String)
    lime_url = Column(String)
    image_url = Column(String)

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )

    user = relationship(
        "User",
        back_populates="analyses"
    )


class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, index=True)

    analysis_id = Column(
        String,
        ForeignKey("analyses.id")
    )

    user_id = Column(
        String,
        ForeignKey("users.id")
    )

    filename = Column(String, nullable=False)
    content = Column(String, nullable=False)

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )

    user = relationship(
        "User",
        back_populates="reports"
    )
