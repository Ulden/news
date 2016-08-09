import json
import re
import requests

r = requests.get('http://api.sports.126.net/api/football/player/playerlist/2016/80.json?jsoncallback=totalplayer')
g = re.match(r'totalplayer\(\n(.+)\)\n', r.content.decode())
playerjson = json.loads(g.group(1))
playerlist = []
for subjson in playerjson:
    for item in playerjson[subjson]:
        playerlist.append(item['player']+'\n')

with open('user_dict.txt', mode = 'w') as f:
    f.writelines(playerlist)
