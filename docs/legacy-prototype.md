# Legacy NFL prototype

FootballVision began as an NFL game-film prototype built around macOS capture,
YOLO detection, ByteTrack, a temporal formation classifier, and simple coach and
fan views. The repository retains that experimental path in `src/pipeline.py`,
`models/classifier.py`, `ui/web_app.py`, and the `/coach` and `/fan` routes.

The actively developed product is now the football (soccer) tactical dashboard
served at `/` and `/tactical`. Its pipeline adds pitch calibration, top-down
projection, team assignment, tactical metrics, local session history, event
review, analytics, and report/video exports.

The legacy code is kept for reference and compatibility. New contributions
should target the tactical pipeline unless an issue explicitly concerns the NFL
prototype.
