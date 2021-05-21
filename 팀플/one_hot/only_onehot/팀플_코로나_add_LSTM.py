'''
SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person,VALUE 
FROM main_data_table AS d 
INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time
LEFT JOIN `covid19_re` AS c ON c.date = d.date

'''

