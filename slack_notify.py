from common import *

###########################################################################################
# token for authentication to send message to slack. You can take it from here: https://api.slack.com/docs/oauth-test-tokens
token = "xoxp-391663010182-390905176885-464369780551-9c1f4b8e5dbb0ee66e7d76ba0c3022e6"
# channel of slack you want to send message
channel = "training_notify"
# name of the Notifier you want to set. This username will be the sender's name in slack
username = "Ngoc Trinh"
# format of the message you want to send. For now I just made it simple with "header", "message" and "color". For more information: https://api.slack.com/docs/oauth-test-tokens
attachments = [{}]
###########################################################################################
Error_list = [
    'Fix me, my hero!',
    'Wake up, something wrong!',
    'Oh god, bugs again?',
]
Warning_list = [
    ''
]
Notify_list = [
    ''
]
###########################################################################################

class SlackNotifier():
    def __init__(self, token=token, channel=channel, username=username):
        self.token = token
        self.channel = channel
        self.username = username
        self.slack_notifier = SlackClient(self.token)

        self.errors = Error_list
        self.warnings = Warning_list
        self.notifies = Notify_list

    def sendNotifyMessage(self, title, message, channel=channel):
        attachments[0]['color'] = "good"
        attachments[0]['title'] = "{} {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"), title)
        attachments[0]['text'] = message
        self.slack_notifier.api_call('chat.postMessage', channel=self.channel, attachments=attachments, username=self.username, icon_emoji=':ngoc-trinh:')

    def sendWarningMessage(self, message, channel=channel):
        attachments[0]['color'] = "warning"
        attachments[0]['title'] = "{} Warning!".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
        attachments[0]['text'] = message
        self.slack_notifier.api_call('chat.postMessage', channel=self.channel, attachments=attachments, username=self.username, icon_emoji=':ngoc-trinh:')

    def sendErrorMessage(self, message, channel=channel):
        attachments[0]['color'] = "danger"
        attachments[0]['title'] = "{} {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"), random.choice(self.errors))
        attachments[0]['text'] =  message
        self.slack_notifier.api_call('chat.postMessage', channel=self.channel, attachments=attachments, username=self.username, icon_emoji=':ngoc-trinh:')