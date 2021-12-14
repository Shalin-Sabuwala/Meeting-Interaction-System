from flask import Flask, render_template, request, jsonify
import track
import os
import time
import DataBaseConnector.mysqlHelper as mysqlHelper

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST']) # To render Homepage
def home_page():
    return render_template('index.html',isProcessed=False)


@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        f = request.files['file']
        print(f)

        # f.save(f.filename)
        return 'file uploaded successfully'
        # return render_template("success.html", name = f.filename)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      dir_path = os.getcwd()
      print(dir_path)
      epochTime =  int(time.time())
      file_name = str(epochTime)+"-"+f.filename
      saveFilePath = dir_path + "/outputFiles/"+file_name
      f.save(saveFilePath)
      process,total_farme = track.InferanceVideo(saveFilePath)
      if process and total_farme>0:
          db = mysqlHelper.MysqlHelper()
          results = db.getProcessData(file_name)
          result_list = [list(elem) for elem in results]
          print(result_list)
          for result in result_list:
              result[2] = os.path.join(result[2])
              videoFPS = (result[4]) / result[3]
              result.append("{:.2f}".format(result[1] * videoFPS))
              result.append("{:.2f}".format((result[1] / result[3]) * 100))

          return render_template("results.html", result=result_list,isProcessed=True,total_farme=total_farme)

      else:
        return render_template("index.html",isProcessed=False)

if __name__ == '__main__':
    PEOPLE_FOLDER = os.path.join('static','inference')
    app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
    print(PEOPLE_FOLDER)
    app.run(debug=True)
