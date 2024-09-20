# Database setup
import logs
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# Database setup
Base = declarative_base()

class Order(Base):
    __tablename__ = 'orders'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    symbol = sqlalchemy.Column(sqlalchemy.String)
    side = sqlalchemy.Column(sqlalchemy.String)
    type = sqlalchemy.Column(sqlalchemy.String)
    quantity = sqlalchemy.Column(sqlalchemy.Float)
    price = sqlalchemy.Column(sqlalchemy.Float)
    timestamp = sqlalchemy.Column(sqlalchemy.DateTime)

class Position(Base):
    __tablename__ = 'positions'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    symbol = sqlalchemy.Column(sqlalchemy.String)
    side = sqlalchemy.Column(sqlalchemy.String)
    quantity = sqlalchemy.Column(sqlalchemy.Float)
    entry_price = sqlalchemy.Column(sqlalchemy.Float)
    exit_price = sqlalchemy.Column(sqlalchemy.Float)
    pnl = sqlalchemy.Column(sqlalchemy.Float)
    timestamp = sqlalchemy.Column(sqlalchemy.DateTime)

engine = sqlalchemy.create_engine('postgresql://user:password@host:port/database')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)