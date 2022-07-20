from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String
from sqlalchemy import create_engine
import numpy as np
from tabulate import tabulate


Base = declarative_base()

class newUsers(Base):
	__tablename__ = 'users'
	#docstring for newUsers
	fname = Column(String(100))
	lname = Column(String(100))
	email = Column(String(30), primary_key = True)
	password = Column(String(20))
	phoneNo = Column(String(15))

engine = create_engine('sqlite:///userregistration.db')
Base.metadata.create_all(engine)
result = engine.execute('SELECT * FROM "users"')
res = result.fetchall()
#print tabulate([res], headers=['Fname', 'Lname','Email','Password','Phoneno'])
# resarr = np.asarray(res)
# print(resarr)