
# coding: utf-8

# ## SQLAlchemy
#
# ##### Author: Daniel Beaulieu | danbeaulieu@gmail.com
import os
from IPython.core.display import display, HTML
from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('../../config.ini')
TEST_DB_PATH = config['DATABASES']['TEST_DB_PATH']
DB_PATH = config['DATABASES']['LESSON_DB_PATH']
STACKOVERFLOW_ZIP_NAME = config['DATABASES']['STACKOVERFLOW_ZIP_NAME']
STACKOVERFLOW_CSV_NAME = config['DATABASES']['STACKOVERFLOW_CSV_NAME']


# SQLAlchemy  SQL toolkit and Object Relational Mapper that
sqlalchemy_url = 'http://www.sqlalchemy.org/'
iframe = '<iframe src={} width=1100 height=300></iframe>'.format(sqlalchemy_url)
HTML(iframe)


# Python's SQLAlchemy and Object-Relational Mapping
#
from sqlalchemy.ext.declarative import declarative_base
# #### Declarative Base
# Declarative base maintains a catalog of classes and tables. It keeps track of all the database tables that are accessible through SQLAlchemy.
Base = declarative_base()

from sqlalchemy import Column, Text, Integer

# provide a class name for the database table
class Sections(Base):
    # provide a name used in SQL to query the table
    __tablename__ = 'SECTIONS'

    # provide column names and data types
    # for each field in the database
    # additional paramaters are available for the columns and data types
    section_id = Column(Integer(), primary_key=True, autoincrement=True)
    section_name = Column(Text())
    section_text = Column(Text())

    def __repr__(self):
        return '<Sections section_id: {} | section_name: {}>'.format(
            self.section_id, self.section_name)


# #### SQLAlchemy Classes
# Each SQLAlchemy class corresponds to a database table.
sections = Sections()
sections
# **_ _repr_ _** (represent) provides the output when using print.

section = Sections(
      section_name = 'first section'
    , section_text = 'text from the first section'
)
section

list_one = ['a','b','c']
list_two = [1,2,3]

print(zip(list_one, list_two))
zip(list_one, list_two)

# confirm test database directory
TEST_DB_PATH
database_url = 'http://docs.sqlalchemy.org/en/latest/core/engines.html'
iframe = '<iframe src={} width=1100 height=300></iframe>'.format(database_url)
HTML(iframe)
from sqlalchemy import create_engine

# create database tables
engine = create_engine(TEST_DB_PATH)
Base.metadata.create_all(engine)

# #### Create Engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(TEST_DB_PATH)
Session = sessionmaker(bind=engine)
session = Session()

# #### Insert data into the database
section = Sections(
      section_name = 'first section'
    , section_text = 'text from the first section'
)

session.add(section)

# commit (save) all annual reports to the database
session.commit()

# query database to check if data was added
result = session.query(Sections).all()
for row in result:
    print('section_id: {}'.format(row.section_id))
    print('section_name: {}'.format(row.section_name))
    print('section_text: {}'.format(row.section_text))

# view the __repr__ of the SQLAlchemy class
result

# Uncomment to delete data from Sections to reset table (optional)
#session.execute("DELETE FROM Sections")
session.commit()



# confirm the path to the stackoverflow database
DB_PATH
# SQLAlchamy Core - SQL Expression Language
from sqlalchemy.sql import text

query = text('SELECT * FROM StackOverflow LIMIT 3')
for result in session.execute(query):
    print('Question Id: {}'.format(result.Id))
    print('TITLE: {}'.format(result.Title))
    print('TAGS: {} \n'.format(result.Tags))

# SQLAlchemy ORM
for result in session.query(StackOverflow)[0:3]:
    print('Question Id: {}'.format(result.Id))
    print('TITLE: {}'.format(result.Title))
    print('TAGS: {} \n'.format(result.Tags))

# view __repr__ of result
print(session.query(StackOverflow)[0])

# Pandas
import pandas as pd
pd.read_sql('SELECT * FROM STACKOVERFLOW LIMIT 3', con=engine)

# SQLAlchamy Core - SQL Expression Language
query = text('SELECT Id, Tags FROM StackOverflow LIMIT 3')
session.execute(query).fetchall()

# SQLAlchemy ORM
session.query(StackOverflow.Id, StackOverflow.Tags).limit(3).all()

# Raw SQL Query
print(session.query(StackOverflow.Id, StackOverflow.Tags).limit(3))

#Pandas
pd.read_sql('SELECT Id, Tags FROM STACKOVERFLOW LIMIT 3', con=engine)


# ### WHERE
# SQLAlchamy Core - SQL Expression Language
query = text('SELECT Id, Tags, Title FROM STACKOVERFLOW WHERE Tags = "python" LIMIT 3')
session.execute(query).fetchall()
# SQLAlchemy ORM
session.query(StackOverflow.Id, StackOverflow.Title
    ).filter(StackOverflow.Tags == 'python').limit(3).all()

# Raw SQL Query
print(session.query(StackOverflow.Id, StackOverflow.Title
    ).filter(StackOverflow.Tags == 'python').limit(3))
# Pandas
pd.read_sql('SELECT Id, Tags, Title FROM STACKOVERFLOW WHERE Tags = "python" LIMIT 3', con=engine)


# ### LIKE
# SQLAlchamy Core - SQL Expression Language
query = text('SELECT Tags FROM STACKOVERFLOW WHERE Tags LIKE "%python%" LIMIT 3')
session.execute(query).fetchall()


# SQLAlchemy ORM
session.query(StackOverflow.Tags
    ).filter(StackOverflow.Tags.like('%python%')).limit(3).all()

# Raw SQL Query
print(session.query(StackOverflow.Tags
    ).filter(StackOverflow.Tags.like('%python%')).limit(3))

# Raw SQL Query
print(session.query(StackOverflow.Tags
    ).filter(StackOverflow.Tags.like('%python%')).limit(3))


# Pandas
pd.read_sql('SELECT Tags FROM STACKOVERFLOW WHERE Tags LIKE "%python%" LIMIT 3', con=engine)

# SQLAlchamy Core - SQL Expression Language
query = text('SELECT Title FROM STACKOVERFLOW WHERE Tags IN ("python","java","sql") LIMIT 3')
session.execute(query).fetchall()


# SQLAlchemy ORM
session.query(StackOverflow.Title
    ).filter(StackOverflow.Tags.in_(['python', 'java', 'sql'])).limit(3).all()


# Raw SQL Query
print(session.query(StackOverflow.Title
    ).filter(StackOverflow.Tags.in_(['python', 'java', 'sql'])).limit(3))

# Pandas
pd.read_sql('SELECT Title FROM STACKOVERFLOW WHERE Tags IN ("python","java","sql") LIMIT 3'
            , con=engine)


# ### COUNT
# SQLAlchamy Core - SQL Expression Language
query = text('SELECT COUNT(*) FROM STACKOVERFLOW')
session.execute(query).scalar()  # scalar returns a single value (no tuple)

# SQLAlchemy ORM
session.query(StackOverflow).count()

# Pandas
pd.read_sql('SELECT COUNT(*) FROM STACKOVERFLOW', con=engine)


# ### GROUP BY
query = text("""SELECT Tags, COUNT(Tags)
            FROM STACKOVERFLOW
            GROUP BY Tags
            ORDER BY COUNT(Tags) Desc
            LIMIT 3""")
session.execute(query).fetchall()  # scalar returns a single value (no tuple)

# SQLAlchemy ORM
from sqlalchemy import func

session.query(StackOverflow.Tags, func.count(StackOverflow.Tags)
    ).group_by(StackOverflow.Tags
    ).order_by(func.count(StackOverflow.Tags).desc()
    ).limit(3).all()

# Raw SQL Query
print(session.query(StackOverflow.Tags, func.count(StackOverflow.Tags)
    ).group_by(StackOverflow.Tags
    ).order_by(func.count(StackOverflow.Tags).desc()
    ).limit(3))

# Pandas
pd.read_sql("""SELECT Tags, COUNT(Tags)
            FROM STACKOVERFLOW
            GROUP BY Tags
            ORDER BY COUNT(Tags) Desc
            LIMIT 3
            """, con=engine)


# ####  load database iteratively from pandas
# confirm dataset is available
print('ZIP NAME: {} \n'.format(STACKOVERFLOW_ZIP_NAME))
print('CSV NAME: {} \n'.format(STACKOVERFLOW_CSV_NAME))
print('DB PATH: {}'.format(DB_PATH))
# NOTE: you must download the stackoverflow zip (2GB zip file) for below code to work
# https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data

import pandas as pd
import datetime as dt
import zipfile
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

zf = zipfile.ZipFile(STACKOVERFLOW_ZIP_NAME)
engine = create_engine(DB_PATH)
for ind, df in enumerate(pd.read_csv(
    zf.open(STACKOVERFLOW_CSV_NAME)
        , chunksize=10_000
        , iterator=True
        , encoding='utf-8'
        , nrows=50_000)):

    # uncomment to write to db
    #df.to_sql('STACKOVERFLOW', con=engine, if_exists='append')
