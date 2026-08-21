import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../../models/capture_session.dart';
import '../../services/db_service.dart';
import '../../services/camera_permission_service.dart';
import '../../services/demo_camera_config.dart';
import '../../services/demo_media_service.dart';
import 'session_saved_screen.dart';

class VideoCaptureScreen extends StatefulWidget {
  final CaptureSession session;

  const VideoCaptureScreen({super.key, required this.session});

  @override
  State<VideoCaptureScreen> createState() => _VideoCaptureScreenState();
}

class _VideoCaptureScreenState extends State<VideoCaptureScreen> {
  CameraController? _controller;

  /// Demo-mode walking-video player (only in Demo Camera Mode).
  VideoPlayerController? _demoVideoController;

  Timer? _timer;
  Future<XFile>? _stopRecordingFuture;

  int _seconds = 0;

  bool _recording = false;
  bool _isCameraReady = false;
  bool _isSaving = false;
  bool _isStopping = false;

  /// True when Demo Camera Mode is active.
  bool get _isDemoMode => DemoCameraConfig.enabled;

  /// True when the photorealistic walking video asset is bundled.
  bool _demoVideoAvailable = false;

  /// Demo-mode live preview image (used as fallback when the walk video is
  /// not bundled).
  String? _demoFallbackAsset;

  // ============================================================
  // INITIALIZE CAMERA / DEMO VIDEO
  // ============================================================

  Future<void> _initializeCamera() async {
    // DEMO MODE: play the realistic walking cow video instead of the camera.
    if (_isDemoMode) {
      await _initializeDemoVideo();
      return;
    }

    try {
      // Ask before touching the camera. Declaring CAMERA in the
      // manifest does not grant it on Android 6+, and without this
      // the preview never appears and nothing reports why.
      final allowed = await CameraPermissionService.ensure(forVideo: true);
      if (!allowed) {
        if (mounted) {
          setState(() {
            _isCameraReady = false;
          });
        }
        return;
      }

      final cameras = await availableCameras();

      if (cameras.isEmpty) {
        if (mounted) {
          _showError('No camera was found on this device.');
        }
        return;
      }

      CameraDescription camera;

      try {
        camera = cameras.firstWhere(
          (camera) => camera.lensDirection == CameraLensDirection.back,
        );
      } catch (_) {
        // Fallback to the first available camera.
        camera = cameras.first;
      }

      final controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await controller.initialize();

      if (!mounted) {
        await controller.dispose();
        return;
      }

      _controller = controller;

      setState(() {
        _isCameraReady = true;
      });
    } on CameraException catch (e) {
      if (kDebugMode) {
        print(
          'Camera initialization error: '
          '${e.code} - ${e.description}',
        );
      }

      if (mounted) {
        _showError(
          'Unable to initialize the camera.\n'
          'Please check camera permissions.',
        );
      }
    } catch (e) {
      if (kDebugMode) {
        print('Unexpected camera initialization error: $e');
      }

      if (mounted) {
        _showError('Unable to initialize the camera.');
      }
    }
  }

  // ============================================================
  // DEMO MODE VIDEO INITIALIZATION
  // ============================================================

  Future<void> _initializeDemoVideo() async {
    final videoKey = DemoCameraConfig.assetKeys[DemoMediaKind.walkingVideo]!;

    final videoExists = await DemoMediaService.assetExists(videoKey);

    if (!mounted) {
      return;
    }

    if (videoExists) {
      final controller = VideoPlayerController.asset(videoKey);

      try {
        await controller.initialize();
      } catch (e) {
        if (kDebugMode) {
          print('Demo video initialization error: $e');
        }

        if (mounted) {
          await controller.dispose();
          setState(() {
            _demoVideoAvailable = false;
            _isCameraReady = true;
          });
        }
        return;
      }

      if (!mounted) {
        await controller.dispose();
        return;
      }

      // Loop so the cow keeps moving if the user waits on this screen.
      await controller.setLooping(true);
      await controller.setVolume(0);

      _demoVideoController = controller;

      setState(() {
        _demoVideoAvailable = true;
        _isCameraReady = true;
      });
      return;
    }

    // Fallback: show the bundled placeholder image instead.
    final fallbackAsset = await DemoMediaService.resolveImageAsset(
      DemoMediaKind.walkingVideo,
    );

    if (!mounted) {
      return;
    }

    setState(() {
      _demoFallbackAsset = fallbackAsset;
      _demoVideoAvailable = false;
      _isCameraReady = true;
    });
  }

  // ============================================================
  // INIT STATE
  // ============================================================

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  // ============================================================
  // DISPOSE
  // ============================================================

  @override
  void dispose() {
    _timer?.cancel();
    _timer = null;

    final demoVideo = _demoVideoController;
    _demoVideoController = null;

    if (demoVideo != null) {
      unawaited(_disposeDemoVideoSafely(demoVideo));
    }

    final controller = _controller;
    _controller = null;
    _isCameraReady = false;

    if (controller != null) {
      unawaited(_disposeControllerSafely(controller));
    }

    super.dispose();
  }

  Future<void> _disposeDemoVideoSafely(VideoPlayerController controller) async {
    try {
      await controller.dispose();
    } catch (_) {
      // Best-effort cleanup during dispose.
    }
  }

  // ============================================================
  // START RECORDING
  // ============================================================

  Future<void> _startRecording() async {
    if (_recording || _isStopping) {
      return;
    }

    // DEMO MODE: restart the walking cow video and start the 8s timer.
    if (_isDemoMode) {
      await _discardPreviousVideo();

      final videoController = _demoVideoController;

      if (videoController != null && _demoVideoAvailable) {
        await videoController.seekTo(Duration.zero);
        await videoController.play();
      }

      if (!mounted) {
        return;
      }

      setState(() {
        _recording = true;
        _isStopping = false;
        _seconds = 0;
      });

      _startTimer();
      return;
    }

    final controller = _controller;

    if (controller == null || !controller.value.isInitialized) {
      _showError('Camera is not ready yet.');
      return;
    }

    if (_recording || _isStopping || controller.value.isRecordingVideo) {
      return;
    }

    try {
      _timer?.cancel();
      _timer = null;
      await _discardPreviousVideo();

      await controller.startVideoRecording();

      if (!mounted) {
        return;
      }

      setState(() {
        _recording = true;
        _isStopping = false;
        _seconds = 0;
      });

      _startTimer();
    } on CameraException catch (e) {
      if (kDebugMode) {
        print(
          'Error starting video recording: '
          '${e.code} - ${e.description}',
        );
      }

      if (mounted) {
        setState(() {
          _recording = false;
        });

        _showError('Unable to start video recording.');
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error starting video recording: $e');
      }

      if (mounted) {
        setState(() {
          _recording = false;
        });

        _showError('Unable to start video recording.');
      }
    }
  }

  // ============================================================
  // START 0 -> 8 SECOND TIMER
  // ============================================================

  void _startTimer() {
    _timer?.cancel();
    _timer = null;

    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }

      setState(() {
        _seconds++;
      });

      if (_seconds >= DemoCameraConfig.videoDurationSeconds) {
        timer.cancel();
        _timer = null;

        // Do not await directly inside Timer callback.
        _finishRecording();
      }
    });
  }

  // ============================================================
  // FINISH RECORDING AT 8 SECONDS
  // ============================================================

  Future<void> _finishRecording() async {
    if (_isStopping) {
      return;
    }

    // DEMO MODE: stop the walking video and copy the bundled demo video into
    // the app temp directory so SQLite receives a valid file path.
    if (_isDemoMode) {
      _isStopping = true;

      try {
        final videoController = _demoVideoController;

        if (videoController != null) {
          await videoController.pause();
          await videoController.seekTo(Duration.zero);
        }

        final result = await DemoMediaService.captureVideo();
        widget.session.videoPath = result.filePath;

        if (kDebugMode) {
          print('Demo video saved to: ${result.filePath}');
        }

        if (!mounted) {
          return;
        }

        setState(() {
          _recording = false;
          _isStopping = false;
          _seconds = DemoCameraConfig.videoDurationSeconds;
        });

        _showRecordingComplete();
      } catch (e) {
        if (kDebugMode) {
          print('Error finishing demo video: $e');
        }

        if (mounted) {
          setState(() {
            _recording = false;
            _isStopping = false;
          });

          _showError('Unable to finish video recording.');
        }
      }
      return;
    }

    final controller = _controller;

    if (controller == null) {
      return;
    }

    if (!controller.value.isInitialized) {
      return;
    }

    if (!controller.value.isRecordingVideo) {
      return;
    }

    try {
      _isStopping = true;
      _timer?.cancel();
      _timer = null;

      _stopRecordingFuture = controller.stopVideoRecording();
      final XFile video = await _stopRecordingFuture!;
      _stopRecordingFuture = null;

      widget.session.videoPath = video.path;

      if (kDebugMode) {
        print('Video saved to: ${video.path}');
      }

      if (!mounted) {
        return;
      }

      setState(() {
        _recording = false;
        _isStopping = false;
        _seconds = 8;
      });

      _showRecordingComplete();
    } on CameraException catch (e) {
      _stopRecordingFuture = null;

      if (kDebugMode) {
        print(
          'Error stopping video recording: '
          '${e.code} - ${e.description}',
        );
      }

      if (mounted) {
        setState(() {
          _recording = false;
          _isStopping = false;
        });

        _showError('Unable to finish video recording.');
      }
    } catch (e) {
      _stopRecordingFuture = null;

      if (kDebugMode) {
        print('Error stopping video recording: $e');
      }

      if (mounted) {
        setState(() {
          _recording = false;
          _isStopping = false;
        });

        _showError('Unable to finish video recording.');
      }
    }
  }

  // ============================================================
  // SHOW RECORDING COMPLETE
  // ============================================================

  void _showRecordingComplete() {
    if (!mounted) {
      return;
    }

    showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          title: const Text('✓ Recording Complete'),
          content: const Text('8-second walking video captured.'),
          actions: <Widget>[
            TextButton(
              onPressed: () {
                Navigator.of(dialogContext).pop();

                // Start a fresh recording.
                _startRecording();
              },
              child: const Text('RE-RECORD'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(dialogContext).pop();

                _saveSession();
              },
              child: const Text('CONTINUE'),
            ),
          ],
        );
      },
    );
  }

  // ============================================================
  // SAVE SESSION TO SQLITE
  // ============================================================

  Future<void> _saveSession() async {
    if (_isSaving) {
      return;
    }

    // Make sure all required media exists.
    if (widget.session.tagId == null || widget.session.tagId!.isEmpty) {
      _showError('Animal ID is missing.');
      return;
    }

    if (widget.session.sidePhotoPath == null ||
        widget.session.sidePhotoPath!.isEmpty) {
      _showError('Side photo is missing.');
      return;
    }

    if (widget.session.rearPhotoPath == null ||
        widget.session.rearPhotoPath!.isEmpty) {
      _showError('Rear photo is missing.');
      return;
    }

    if (widget.session.videoPath == null || widget.session.videoPath!.isEmpty) {
      _showError('Walking video is missing.');
      return;
    }

    setState(() {
      _isSaving = true;
    });

    try {
      // The local id is what lets the next screen push this capture to the
      // server straight away and come back with a scorecard, instead of the
      // farmer being told "saved" and never seeing a result.
      final localId = await DbService.insertSession(widget.session);

      if (!mounted) {
        return;
      }

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => SessionSavedScreen(
            localId: localId,
            sidePhotoPath: widget.session.sidePhotoPath,
          ),
        ),
      );
    } catch (e) {
      if (kDebugMode) {
        print('Error saving session: $e');
      }

      if (mounted) {
        setState(() {
          _isSaving = false;
        });

        _showError(
          'Unable to save the session.\n'
          'Please try again.',
        );
      }
    }
  }

  Future<void> _discardPreviousVideo() async {
    final oldPath = widget.session.videoPath;
    widget.session.videoPath = null;

    if (oldPath == null || oldPath.isEmpty) {
      return;
    }

    try {
      final file = File(oldPath);
      if (await file.exists()) {
        await file.delete();
      }
    } catch (_) {
      // Temporary cleanup failure should not block a re-record.
    }
  }

  Future<void> _disposeControllerSafely(CameraController controller) async {
    try {
      final pendingStop = _stopRecordingFuture;

      if (pendingStop != null) {
        try {
          await pendingStop;
        } catch (_) {
          // The screen is closing; a failed in-flight stop should not escape.
        }
      } else if (controller.value.isInitialized &&
          controller.value.isRecordingVideo) {
        try {
          _isStopping = true;
          await controller.stopVideoRecording();
        } catch (_) {
          // Best-effort cleanup during dispose.
        }
      }
    } finally {
      try {
        await controller.dispose();
      } catch (_) {
        // Best-effort cleanup during dispose.
      }
    }
  }

  // ============================================================
  // ERROR DIALOG
  // ============================================================

  void _showError(String message) {
    if (!mounted) {
      return;
    }

    showDialog<void>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Error'),
          content: Text(message),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: const Text('OK'),
            ),
          ],
        );
      },
    );
  }

  // ============================================================
  // BUILD
  // ============================================================

  @override
  Widget build(BuildContext context) {
    if (!_isCameraReady) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator()),
      );
    }

    // DEMO MODE: the walking cow video is the "camera feed".
    if (_isDemoMode) {
      return _buildDemoBody();
    }

    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Step 4: Walking Video'),
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: Stack(
        children: [
          // ======================================================
          // CAMERA PREVIEW
          // ======================================================
          Positioned.fill(child: CameraPreview(_controller!)),

          // ======================================================
          // RECORDING OVERLAY
          // ======================================================
          if (_recording)
            Positioned.fill(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text(
                    'CAMERA',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  const SizedBox(height: 20),

                  const Text(
                    'WALK NOW',
                    style: TextStyle(
                      color: Colors.redAccent,
                      fontSize: 30,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  const SizedBox(height: 40),

                  // ==================================================
                  // COUNT-UP TIMER
                  // ==================================================
                  Text(
                    '$_seconds',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 60,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  const Text(
                    'seconds',
                    style: TextStyle(color: Colors.white, fontSize: 20),
                  ),

                  const SizedBox(height: 30),

                  const Text(
                    'Recording automatically stops at 8 seconds',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                ],
              ),
            )
          // ======================================================
          // START RECORDING BUTTON
          // ======================================================
          else
            Positioned(
              bottom: 35,
              left: 0,
              right: 0,
              child: GestureDetector(
                onTap: _startRecording,
                child: Column(
                  children: [
                    Container(
                      width: 72,
                      height: 72,
                      decoration: BoxDecoration(
                        color: Colors.redAccent,
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.grey, width: 4),
                      ),
                      child: const Icon(
                        Icons.videocam,
                        color: Colors.white,
                        size: 32,
                      ),
                    ),

                    const SizedBox(height: 8),

                    const Text(
                      'START RECORDING (8s)',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),

          // ======================================================
          // SAVING OVERLAY
          // ======================================================
          if (_isSaving)
            Positioned.fill(
              child: Container(
                color: Colors.black54,
                child: const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      CircularProgressIndicator(color: Colors.white),
                      SizedBox(height: 20),
                      Text(
                        'Saving session...',
                        style: TextStyle(color: Colors.white, fontSize: 18),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  // ============================================================
  // DEMO MODE BODY — walking cow video behind the existing UI
  // ============================================================

  Widget _buildDemoBody() {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Step 4: Walking Video'),
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: Stack(
        children: [
          // ======================================================
          // DEMO COW FEED
          // ======================================================
          Positioned.fill(child: _buildDemoFeed()),

          // ======================================================
          // RECORDING OVERLAY
          // ======================================================
          if (_recording)
            Positioned.fill(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text(
                    'CAMERA',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  const SizedBox(height: 20),

                  const Text(
                    'WALK NOW',
                    style: TextStyle(
                      color: Colors.redAccent,
                      fontSize: 30,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  const SizedBox(height: 40),

                  // ==================================================
                  // COUNT-UP TIMER
                  // ==================================================
                  Text(
                    '$_seconds',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 60,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  const Text(
                    'seconds',
                    style: TextStyle(color: Colors.white, fontSize: 20),
                  ),

                  const SizedBox(height: 30),

                  const Text(
                    'Recording automatically stops at 8 seconds',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                ],
              ),
            )
          // ======================================================
          // START RECORDING BUTTON
          // ======================================================
          else
            Positioned(
              bottom: 35,
              left: 0,
              right: 0,
              child: GestureDetector(
                onTap: _startRecording,
                child: Column(
                  children: [
                    Container(
                      width: 72,
                      height: 72,
                      decoration: BoxDecoration(
                        color: Colors.redAccent,
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.grey, width: 4),
                      ),
                      child: const Icon(
                        Icons.videocam,
                        color: Colors.white,
                        size: 32,
                      ),
                    ),

                    const SizedBox(height: 8),

                    const Text(
                      'START RECORDING (8s)',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),

          // ======================================================
          // SAVING OVERLAY
          // ======================================================
          if (_isSaving)
            Positioned.fill(
              child: Container(
                color: Colors.black54,
                child: const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      CircularProgressIndicator(color: Colors.white),
                      SizedBox(height: 20),
                      Text(
                        'Saving session...',
                        style: TextStyle(color: Colors.white, fontSize: 18),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildDemoFeed() {
    final videoController = _demoVideoController;

    if (videoController != null && _demoVideoAvailable) {
      return Center(
        child: AspectRatio(
          aspectRatio: videoController.value.aspectRatio,
          child: VideoPlayer(videoController),
        ),
      );
    }

    // Fallback: static cow image when the walk video asset is not bundled.
    final asset = _demoFallbackAsset;
    if (asset != null) {
      return Image.asset(asset, fit: BoxFit.cover);
    }

    return const ColoredBox(color: Colors.black);
  }
}
