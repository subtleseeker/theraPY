from flask import Flask, render_template
import os,subprocess
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/my-link/')
def my_link():
  # os.system("script.py 1")
  print ('Lets do it!')
  subprocess.call('python3 script.py 1', shell=True)
  return 'Click.'

if __name__ == '__main__':
  app.run(debug=True)