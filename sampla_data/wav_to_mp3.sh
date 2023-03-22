for f in *.mp3; do ffmpeg -i "$f" -acodec pcm_s16le -ac 1 -ar 16000 "${f%%.*}.wav"; done 
mkdir clips_wav 
find . -name '*.wav' | xargs mv --target-directory=clips_wav

