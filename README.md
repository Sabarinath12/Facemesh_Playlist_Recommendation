# Facemesh Playlist Recommendation

A computer vision application that detects your facial emotions in real-time and automatically plays matching Spotify playlists using MediaPipe's facial landmark detection.

## Features

- Real-time emotion detection (happy, sad, angry, surprised, neutral)
- Automatic Spotify playlist opening based on detected emotions
- Emotion smoothing to prevent rapid switching
- Visual feedback with confidence levels

## Requirements

```bash
pip install opencv-python mediapipe numpy
```

- Python 3.7+
- Webcam
- Spotify installed

## Usage

1. Run the application:
   ```bash
   python facemesh_playlist_recommendation.py
   ```

2. Position yourself in front of the camera with good lighting

3. Express emotions naturally - when consistently detected (60%+ confidence), matching Spotify playlists will open

4. Press 'q' to quit

## Customization

Modify `music_config` in the code to use your own Spotify playlist IDs:

```python
self.music_config = {
    "happy": ["your_playlist_id_1", "your_playlist_id_2"],
    "sad": ["your_playlist_id_3"],
    # ... add your playlist IDs
}
```

Find playlist IDs from Spotify share links: `https://open.spotify.com/playlist/[PLAYLIST_ID]`

## Troubleshooting

- **Camera issues**: Try `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`
- **Spotify not opening**: Ensure Spotify is installed and test playlist URLs manually
- **Poor detection**: Improve lighting and ensure face is clearly visible
