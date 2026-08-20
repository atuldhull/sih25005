# Cow Demo Media — Demo Camera Mode

Demo Camera Mode renders realistic bundled cow media instead of a live camera.

When you are ready for a photorealistic demo, add these files to this folder:

| File(s)             | Purpose                                  |
|---------------------|------------------------------------------|
| `cow_tag.jpg`       | Head/ear close-up with the ear tag visible |
| `cow_side.jpg`      | Full body side profile                     |
| `cow_rear.jpg`      | Full body rear view                        |
| `cow_walking.mp4`   | 8–10 second walking-video                  |

Guidelines from the project brief:

- Photorealistic, natural lighting, Indian dairy/farm cow
- Same cow and environment across all four files
- Natural phone-camera framing like a field worker capture
- No UI, no text, no watermarks, no cartoon / 3D render look
- `cow_walking.mp4` should visibly show the cow walking for the full
  8-second recording window

Until the files above are present, the app automatically falls back to the
bundled placeholder image (`assets/demo/cow_ear_tag.png`) so the demo capture
flow still runs end to end.