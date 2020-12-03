# @Author: UnsignedByte
# @Date:   21:30:07, 02-Dec-2020
# @Last Modified by:   UnsignedByte
# @Last Modified time: 21:30:23, 02-Dec-2020

# create venv if not exist
if [ ! -d "./.venv" ]; then
  virtualenv --python=python3 --system-site-packages .venv
fi


# activate venv
source .venv/bin/activate
pip install -r requirements.txt