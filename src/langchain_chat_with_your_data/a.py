from googleapiclient.discovery import build
import os

api_key = os.environ['YOUTUBE_API_KEY'] # You need to have a YouTube API key for this to work
youtube = build('youtube', 'v3', developerKey=api_key)

request = youtube.search().list(
    part='snippet',
    maxResults=10,
    q='horror movie', # You can replace this with any search query
    type='video'
)
response = request.execute()

for item in response['items']:
    print(item['snippet']['title'])
