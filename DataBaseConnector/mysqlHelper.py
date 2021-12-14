import mysql.connector
import numpy as np

class MysqlHelper:
  mydb = None
  def __init__(self):
    self.mydb = mysql.connector.connect(host='127.0.0.1',user="root",password="root",database="eye_detection")
    # self.mycursor = mydb.cursor()


  def saveData(self,is_focus,frame_number,people_id,unique_token,video_name,image_file_path,total_farme,total_time):
    sql = "INSERT INTO detection_data (is_focus, frame_number,people_id,token,video_name,image_file_path,total_farme,total_time) VALUES (%s, %s,%s,%s,%s,%s,%s,%s)"
    # print(int(is_focus))
    val = (int(is_focus), frame_number,int(str(people_id)),unique_token,video_name,image_file_path,total_farme,total_time)
    self.mydb.cursor().execute(sql, val)
    self.mydb.commit()

  def getProcessData(self,video_name):
    sql = "SELECT people_id,count(id),image_file_path,total_farme,total_time FROM detection_data where is_focus = 1 and " \
          +"video_name = '{}' group by people_id ,is_focus,image_file_path,total_farme,total_time ORDER BY people_id ASC ".format(video_name)
    cursor = self.mydb.cursor(buffered=True)
    cursor.execute(sql)
    op = cursor.fetchall()
    return op