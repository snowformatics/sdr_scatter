from datetime import date
import os

path = "E:/sdr/meteors/positives/"
l = os.listdir(path)
today = date.today()
print("Today's date:", today)
f = open('test2.html', 'w')
f.write(
"""<html>
<head>
 <title> Cell comparator</title>
</head>
<body>

Meteor Waterfall

</body>
</html>

<table border="1"  width="800">
<tr><th>Picture1</th><th>R75</th><th>Picture2</th></tr>
""")

for i in l:
    if i.endswith('.jpg'):
        f.write("""<tr> <td align = "right"> <img src = """ + '"' + i + '"' + """/> </td>  </tr>""")


