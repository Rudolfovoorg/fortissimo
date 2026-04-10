print("we are at models")
from sqlalchemy import ForeignKey
from sqlalchemy import  Column, Integer, DateTime, BigInteger, Float
from sqlalchemy import String
from sqlalchemy.orm import Session
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from typing import List
from typing import Optional


class Base(DeclarativeBase):

    pass
class Clients(Base):
    __tablename__="Clients"
    Id = Column(Integer, primary_key=True)  # Note: using 'Id' to match your schema
    # Plant details
    Longitude = Column(Float, comment="double precision")
    Latitude = Column(Float, comment="double precision") 
    Type = Column(Integer, comment="integer")
    Code = Column(String, comment="text or varchar")  # Adjust length as needed
    PlantGridPower = Column(Float, comment="double precision")
    PlantPvPower = Column(Float, comment="double precision")
    PlantStorageNominalCap = Column(Float, comment="double precision")
    PlantStoragePower = Column(Float, comment="double precision")
    plantId = Column(Integer, comment="integer")  # This seems to be a separate field
    EnergyData: Mapped[List["EnergyData"]]=relationship("EnergyData", back_populates="Client")
    Measurements: Mapped[List["Measurements"]]=relationship("Measurements", back_populates="Client")
    PVPredictions :Mapped[List["PVProductionPrediction"]] = relationship("PVProductionPrediction", back_populates="Client")
    

class LoadPrediction(Base):
    __tablename__ = 'LoadPredictions'

    Id = Column(Integer, primary_key=True, autoincrement=True) 
    Time = Column(DateTime, nullable=False)        # Čas napovedi
    ClientId = Column(Integer, nullable=False)
    LSTMLoadPower = Column(BigInteger, nullable=False) # W
    HybridLoadPower = Column(BigInteger, nullable=False) # W
    GridPower = Column(BigInteger, nullable=True)  # W

class LoadPrediction1h(Base):
    __tablename__ = 'LoadPredictions1h'
    Id = Column(Integer, primary_key=True, autoincrement=True) 
    Time = Column(DateTime, nullable=False)        # Čas napovedi
    ClientId = Column(Integer, nullable=False)     # ID stranke
    LoadPower = Column(BigInteger, nullable=False) # W
    GridPower = Column(BigInteger, nullable=True)  # W

class ElectricityTariffs(Base):
    __tablename__ = 'tariff_schedules'
    Id = Column(Integer, primary_key=True)
    ScheduleName=Column(String)
    TimeBlocks= Column(String)

class PVProductionPrediction(Base):
    __tablename__= 'PVProductionPrediction'
    Time = Column(DateTime, nullable=False, primary_key=True)
    ClientId = Column(Integer, ForeignKey('Clients.Id'), nullable=False, primary_key=True)
    PVProductionPower = Column(BigInteger, nullable=False)
    Client = relationship("Clients", back_populates="PVPredictions")

class EnergyData(Base):
    __tablename__ = 'EnergyData'  # Replace with your actual table name    
    # Primary key (you may need to add this if not present)
    TimeStampMeasured = Column(DateTime(timezone=True),primary_key=True, comment="timestamp with time zone")
    ClientId = Column(Integer,ForeignKey(Clients.Id), comment="integer",primary_key=True)
    CycleDay = Column(Integer, comment="integer") 
    CycleMinute = Column(Integer, comment="integer")
    BlockNumber = Column(Integer, comment="integer")
    PowerToGrid = Column(BigInteger, comment="bigint")
    PowerFromGrid = Column(BigInteger, comment="bigint")
    PowerFromLoad = Column(BigInteger, comment="bigint")
    PowerToPv =Column(BigInteger, comment="bigint",nullable=True)
    PowerFromPv =Column(BigInteger, comment="bigint",nullable=True)
    EnergyToGrid = Column(BigInteger, comment="bigint")
    EnergyFromGrid = Column(BigInteger, comment="bigint")
    EnergyToLoad = Column(BigInteger, comment="bigint")
    EnergyFromLoad = Column(BigInteger, comment="bigint")
    EnergyToStorage = Column(BigInteger, comment="bigint")
    EnergyFromStorage = Column(BigInteger, comment="bigint")
    EnergyToPv = Column(BigInteger, comment="bigint")
    EnergyFromPv = Column(BigInteger, comment="bigint")
    StorageSocPercentage = Column(Integer, comment="integer")
    StorageSocKwh = Column(Float, comment="double precision")
    EnergyGridToCounter= Column(BigInteger, comment="bigint")
    EnergyBatteryFromCounter= Column(BigInteger, comment="bigint")
    LoadEnergyCalculated =Column(Float, comment="float")
    EnergyPvCounter= Column(BigInteger, comment="bigint")
    Client = relationship("Clients",back_populates="EnergyData")


class Measurements(Base):
    __tablename__ = 'Measurements'
    
    # Primary key (assuming Time + ClientId as composite key, or add an ID field)
    Time = Column(DateTime(timezone=True), primary_key=True)
    ClientId = Column(Integer,ForeignKey(Clients.Id), comment="integer",primary_key=True)
    # Power measurements
    PowerHEE = Column(Float)
    PowerSetpointHEE = Column(Float)
    PowerGrid = Column(Float)
    PowerPV = Column(Float)
    PowerConsumer = Column(Float)
    # Battery/Storage
    SoC = Column(Integer)
    SoCKWH = Column(Float)
    # Weather data
    Weather = Column(String)
    Temperature = Column(Float)
    Clouds = Column(Integer)
    Visibility = Column(Integer)
    SunriseDT = Column(DateTime(timezone=True))
    SunsetDT = Column(DateTime(timezone=True))
    WeatherDescription = Column(String)
    Pressure = Column(Integer)
    Humidity = Column(Integer)
    WindSpeed = Column(Float)
    Client = relationship("Clients",back_populates="Measurements")

class Predictions(Base):
    __tablename__ = "Predictions"
    
    # Composite primary key (similar to EnergyData)
    Time = Column(DateTime(timezone=True), primary_key=True, comment="timestamp with time zone")
    ClientId = Column(Integer, ForeignKey(Clients.Id), primary_key=True, comment="integer")
    GridPrice = Column(Float, comment="double precision")
    SoCKWH = Column(Float, comment="double precision")
    PowerSetpointHEE = Column(Float, comment="double precision")
    MinimumSoC = Column(Float, comment="double precision")
    MaximumSoC = Column(Float, comment="double precision")
    Mode = Column(Integer, comment="integer")
    Client = relationship("Clients")
