import smtplib

gmail_user = 'kendallweihe@gmail.com'
gmail_password = 'B0bbleh3adjoe'

try:
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(gmail_user, gmail_password)
    server.sendmail("kendallweihe@gmail.com", "5022165761@vtext.com", "testing")
except:
    print 'Something went wrong...'
