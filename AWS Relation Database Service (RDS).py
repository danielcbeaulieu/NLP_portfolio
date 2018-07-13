
# coding: utf-8

# # AWS Relational Database Service with Python
#
# ##### Author: Daniel Beaulieu | danbeaulieu@gmail.com

# In[48]:


import os
from IPython.core.display import display, HTML
from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('../../config.ini')

STACKOVERFLOW_ZIP_NAME = config['DATABASES']['STACKOVERFLOW_ZIP_NAME']
STACKOVERFLOW_CSV_NAME = config['DATABASES']['STACKOVERFLOW_CSV_NAME']
DB_PATH = 'mysql://USERNAME:PASSWORD@AWS_ENDPOINT/DB_NAME'

# sqlalchemy imports
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Text, Integer
from sqlalchemy.sql import text
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import reflection
import pandas as pd

# ### AWS RDS (MySQL)
#
# ##### Installation:
# - pip install mysqlclient

DB_PATH

# instantiate Base
Base = declarative_base()

# define database table
class Sections(Base):
    __tablename__ = 'SECTIONS'

    section_id =   Column(Integer(), primary_key=True, autoincrement=True)
    section_name = Column(Text())
    section_text = Column(Text())

# create database table
engine = create_engine(DB_PATH)
Base.metadata.create_all(engine)

# create a session - connect to db
engine = create_engine(DB_PATH)
Session = sessionmaker(bind=engine)
session = Session()

# create a record
section = Sections(
      section_name = 'first section'
    , section_text = 'text from the first section'
)

# add and commit the record into the database
session.add(section)
session.commit()

# query database to check if data was added
pd.read_sql('SELECT * FROM SECTIONS', con=engine)

# Optional - Delete Table
session.execute("DELETE FROM SECTIONS")
session.commit()


# ### Load Existing Dataset to AWS RDS (MySQL)

# confirm dataset is available
print('ZIP NAME: {} \n'.format(STACKOVERFLOW_ZIP_NAME))
print('CSV NAME: {}'.format(STACKOVERFLOW_CSV_NAME))

Base = declarative_base()

class StackOverflow(Base):
    __tablename__ = 'STACKOVERFLOW'
    Id =    Column(Integer(),  primary_key=True,nullable=False)
    Title = Column(Text(), nullable=True)
    Body =  Column(Text(), nullable=True)
    Tags =  Column(Text(), nullable=True)

engine = create_engine(DB_PATH)
Base.metadata.create_all(engine)

import pandas as pd
import datetime as dt
import zipfile

start = dt.datetime.now() # set start time
chunksize = 10000  # set number of row to load at a time
engine = create_engine(DB_PATH)  # connect to database
zf = zipfile.ZipFile(STACKOVERFLOW_ZIP_NAME)  # open zipfile

for ind, df in enumerate(pd.read_csv(zf.open(STACKOVERFLOW_CSV_NAME)
    , chunksize=chunksize
    , iterator=True
    , encoding='latin-1'
    , nrows=50000)):

    # print metrics (time/row) for populating database
    print('{} seconds: completed {} rows'.format(
        (dt.datetime.now() - start).seconds, ind*chunksize))

    # incrementally load the database
    df.to_sql('STACKOVERFLOW', con=engine, if_exists='append', index=False)

# connect to the database
engine = create_engine(DB_PATH)
Session = sessionmaker(bind=engine)
session = Session()

# query
df = pd.read_sql('SELECT * FROM STACKOVERFLOW', con=engine)
df.head()




# Optional - Delete Table
query = text('DROP TABLE STACKOVERFLOW')
#session.execute(query)


# #### Determine the schema of an existing database

# view all TABLES that exist in the database
query = text('SHOW TABLES')
print('TABLES: {} \n'.format(session.execute(query).fetchall()))

# view the field names of a specific table
insp = reflection.Inspector.from_engine(engine)
for col in insp.get_columns('STACKOVERFLOW'):
    print('name: {} | type: {} | nullable: {}'.format(col['name'], col['type'], col['nullable']))
