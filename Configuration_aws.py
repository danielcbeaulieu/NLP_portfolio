
# coding: utf-8

# # Configuration
#
# ##### Author: Daniel Beaulieu @ danbeaulieu@gmail.com

# ### ConfigParser
from IPython.core.display import display, HTML
import configparser

# instantiate config parser
config = configparser.ConfigParser()

# read a config file
config.read('config1.ini')

# observe the sections in the config file
config.sections()

# config uses dict syntax to store values
config['default']['region']

# store all config values
REGION = config['default']['region']
OUTPUT = config['default']['region']
AWS_ACCESS_KEY_ID = config['keys']['aws_access_key_id']
AWS_SECRET_ACCESS_KEY = config['keys']['aws_secret_access_key']

# import, instantiate, and read a config parser with extended interpolation
from configparser import ConfigParser, ExtendedInterpolation

config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
config.read('config_extended.ini')

config.sections()

# view project config file
get_ipython().run_line_magic('less', '../../../config.ini')
