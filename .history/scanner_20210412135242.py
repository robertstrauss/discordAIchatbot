import discord
import re
import sys, os
import requests
import random
import time
import asyncio
import pandas as pd

client = discord.Client()

guildID = 755473404643115170

guild = None

channelID = 831188849946394654

talkchannel = None

# async def pingean():
#     while True:

#         await asyncio.sleep(pinginterval)
#         print(pinging)
#         if (pinging):
#             role = random.choice(ean.roles)
#             await pingchannel.send('{}'.format(role.mention))


# def startpingean():
#     global pinging
#     pinging = True

# def stoppingean():
#     global pinging
#     pinging = False

transcriptsfolder = 'channeltranscripts/'


@client.event
async def on_ready():
    global guild, pingchannel, ean, logan, sam
    print('We have logged in as {0.user}'.format(client))
    guild = client.get_guild(guildID)
    # talkchannel = client.get_channel(channelID)

    limit = int(re.findall(r'limit=(\d+)', " ".join(sys.argv))[0])

    channelIDs = re.findall(r' (\d+) ', " ".join(sys.argv))

    if len(channelIDs) == 0:
        channelIDs = [channel.id for channel in guild.channels]

    for channelID in channelIDs:
        channelID = int(channelID)
        channel = client.get_channel(channelID)

        if type(channel) == discord.channel.TextChannel:
            print('scanning', channel)
            messages = await collectmessages(channel, limit)
            
            category = guild.get_channel(channel.category_id)
            if type(category) == discord.channel.CategoryChannel:
                categoryname = category.name
            else:
                categoryname = '.'

            foldername = os.path.join(transcriptsfolder, categoryname)

            
            os.makedirs(foldername, exist_ok=True)
            # print(data)
            with open(os.path.join(foldername, channel.name+'.txt'), 'w') as channeltxt:
                channeltxt.write('\n'.join(messages))
            # data.to_csv('{}/{}.csv'.format(folder, channel.name))


async def collectmessages(channel, limit):
    # data = pd.DataFrame(columns=['content', 'author'])
    messages = []
    lastauthor = ''

    try:
        # hist = channel.history(limit=limit)
        async for msg in channel.history(limit=limit):
            print('msg', msg.content, msg.author)
            

            print(msg.author, client.user)
            if msg.author != client.user:
                print('bruuuuhh')
                content = msg.content.replace('\n', ' ')

                # print('-1', data[-1], data[-1].author, data.author, data.author[-1])
                if msg.author == lastauthor:
                    messages[-1] = content + messages[-1]
                    print('merged')
                else:
                    # data = data.append({'content':  content,
                                        # 'time':     msg.created_at,
                                        # 'author':   msg.author.name},
                                        # ignore_index=True)
                    messages.prepend(content)
                    print('prepended')
                lastauthor = msg.author
            # if len(messages) == limit:
            #     break
    except (discord.errors.Forbidden):
        print('access denied to', channel.name)
    except Exception as e:
        raise e
    finally:
        return messages




with open('token.txt', 'r') as tokentxt:
    # asyncio.get_event_loop().create_task(pingean())
    client.run(tokentxt.read())
