import 'dart:io';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import '../../models/capture_session.dart';
import '../../services/camera_permission_service.dart';
import '../../services/demo_camera_config.dart';
import '../../services/demo_media_service.dart';
import '../../widgets/rear_silhouette_painter.dart';
import 'photo_quality_screen.dart';
import 'video_capture_screen.dart';

class RearPhotoScreen extends StatefulWidget {
  final CaptureSession session;

  const RearPhotoScreen({super.key, required this.session});
  @override
  State<RearPhotoScreen> createState() => _RearPhotoScreenState();
}

class _RearPhotoScreenState extends State<RearPhotoScreen> {
  CameraController? _controller;
  bool _isCapturing = false;
  bool _isDisposingCamera = false;
  String? _cameraError;

  /// Demo-mode live preview asset (resolved once at startup).
  String? _demoPreviewAsset;

  // ============================================
  // INITIALIZE THE BACK CAMERA
  // ============================================
  Future<void> _initializeCamera() async {
    // DEMO MODE: no real camera is initialized. The realistic rear-view cow
    // image is shown behind the existing silhouette guide.
    if (DemoCameraConfig.enabled) {
      final asset = await DemoMediaService.resolveImageAsset(
        DemoMediaKind.rear,
      );

      if (!mounted) return;

      setState(() {
        _demoPreviewAsset = asset;
        _cameraError = null;
      });
      return;
    }

    try {
      // Ask before touching the camera. Declaring CAMERA in the
      // manifest does not grant it on Android 6+, and without this
      // the preview never appears and nothing reports why.
      final allowed = await CameraPermissionService.ensure(forVideo: false);
      if (!allowed) {
        if (mounted) {
          setState(() {
            _cameraError = 'Camera access is needed to capture the rear photo. Allow it in Settings and try again.';
          });
        }
        return;
      }

      final cameras = await availableCameras();

      if (cameras.isEmpty) {
        if (mounted) {
          setState(() {
            _cameraError = 'No camera was found on this device.';
          });
        }
        return;
      }

      final camera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

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
        _cameraError = null;
      });
    } on CameraException catch (e) {
      if (mounted) {
        setState(() {
          _cameraError = _cameraMessage(e);
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _cameraError = 'Unable to initialize the camera.';
        });
      }
    }
  }

  // ============================================
  // INITIALIZE CAMERA WHEN SCREEN OPENS
  // ============================================
  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  // ============================================
  // RELEASE CAMERA WHEN SCREEN CLOSES
  // ============================================
  @override
  void dispose() {
    _controller?.dispose();
    _controller = null;
    super.dispose();
  }

  // ============================================
  // CAPTURE PHOTO
  // ============================================
  Future<void> _capturePhoto() async {
    if (_isCapturing || _isDisposingCamera) {
      return;
    }

    // DEMO MODE: copy the bundled rear-view cow image into a real temp file
    // and run the existing quality gate unchanged.
    if (DemoCameraConfig.enabled) {
      setState(() => _isCapturing = true);

      try {
        final result = await DemoMediaService.captureImage(DemoMediaKind.rear);

        if (mounted) {
          await _runQualityGate(result.filePath);
        }
      } catch (_) {
        if (mounted) {
          _showError('Unable to capture the rear photo.');
        }
      }

      if (mounted) {
        setState(() => _isCapturing = false);
      }
      return;
    }

    final controller = _controller;

    if (controller == null ||
        !controller.value.isInitialized ||
        _isCapturing ||
        _isDisposingCamera) {
      return;
    }

    setState(() => _isCapturing = true);

    try {
      final XFile photo = await controller.takePicture();

      if (mounted) {
        await _runQualityGate(photo.path);
      }
    } on CameraException catch (e) {
      if (mounted) {
        _showError(_cameraMessage(e));
      }
    } catch (_) {
      if (mounted) {
        _showError('Unable to capture the rear photo.');
      }
    }

    if (mounted) {
      setState(() => _isCapturing = false);
    }
  }

  // ============================================
  // RUN QUALITY GATE
  // ============================================
  Future<void> _runQualityGate(String path) async {
    final accepted = await Navigator.push<bool>(
      context,
      MaterialPageRoute(builder: (_) => PhotoQualityScreen(imagePath: path)),
    );

    if (!mounted) return;

    if (accepted == true) {
      widget.session.rearPhotoPath = path;

      await _disposeCameraForNavigation();

      if (!mounted) return;

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => VideoCaptureScreen(session: widget.session),
        ),
      );
      return;
    }

    widget.session.rearPhotoPath = null;
    await _deleteTemporaryFile(path);
  }

  Future<void> _disposeCameraForNavigation() async {
    if (_isDisposingCamera) {
      return;
    }

    _isDisposingCamera = true;

    final controller = _controller;
    _controller = null;

    try {
      await controller?.dispose();
    } catch (_) {
      // Camera release failure should not block moving to the next step.
    }
  }

  Future<void> _deleteTemporaryFile(String path) async {
    try {
      final file = File(path);
      if (await file.exists()) {
        await file.delete();
      }
    } catch (_) {
      // Temporary cleanup failure should not block a retake.
    }
  }

  String _cameraMessage(CameraException e) {
    if (e.code.toLowerCase().contains('permission')) {
      return 'Camera permission is required to capture the rear photo.';
    }

    return 'Unable to use the camera. Please check permissions and try again.';
  }

  void _showError(String message) {
    showDialog<void>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Camera Error'),
          content: Text(message),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('OK'),
            ),
          ],
        );
      },
    );
  }

  // ============================================
  // BUILD CAMERA SCREEN
  // ============================================
  @override
  Widget build(BuildContext context) {
    // Show loading indicator while camera initializes
    if (_cameraError != null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Step 3: Rear View')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Text(
              _cameraError!,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 18),
            ),
          ),
        ),
      );
    }

    // DEMO MODE: show the realistic rear-view cow image behind the existing
    // silhouette guide and capture button.
    if (DemoCameraConfig.enabled) {
      final asset = _demoPreviewAsset;
      if (asset == null) {
        return const Scaffold(body: Center(child: CircularProgressIndicator()));
      }

      return Scaffold(
        appBar: AppBar(
          title: const Text('Step 3: Rear View'),
          backgroundColor: Colors.transparent,
          foregroundColor: Colors.white,
          elevation: 0,
        ),
        backgroundColor: Colors.black,
        body: Stack(
          children: [
            // ============================================
            // DEMO COW PREVIEW
            // ============================================
            Positioned.fill(child: Image.asset(asset, fit: BoxFit.cover)),

            // ============================================
            // SILHOUETTE GOES HERE
            // ============================================
            Center(
              child: CustomPaint(
                size: const Size(300, 450),
                painter: RearSilhouettePainter(),
              ),
            ),

            // ============================================
            // CAPTURE BUTTON GOES HERE
            // ============================================
            Positioned(
              bottom: 35,
              left: 0,
              right: 0,
              child: GestureDetector(
                onTap: _capturePhoto,
                child: Column(
                  children: [
                    Container(
                      width: 72,
                      height: 72,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.grey, width: 4),
                      ),
                      child: _isCapturing
                          ? const CircularProgressIndicator()
                          : const Icon(
                              Icons.camera_alt,
                              color: Colors.black,
                              size: 32,
                            ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _isCapturing ? 'CAPTURING...' : 'CAPTURE REAR VIEW',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      );
    }

    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    // Fullscreen camera preview using Stack
    return Scaffold(
      appBar: AppBar(
        title: const Text('Step 3: Rear View'),
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // ============================================
          // CAMERA PREVIEW
          // ============================================
          Positioned.fill(child: CameraPreview(_controller!)),

          // ============================================
          // SILHOUETTE GOES HERE
          // ============================================
          Center(
            child: CustomPaint(
              size: const Size(300, 450),
              painter: RearSilhouettePainter(),
            ),
          ),

          // ============================================
          // CAPTURE BUTTON GOES HERE
          // ============================================
          Positioned(
            bottom: 35,
            left: 0,
            right: 0,
            child: GestureDetector(
              onTap: _capturePhoto,
              child: Column(
                children: [
                  Container(
                    width: 72,
                    height: 72,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                      border: Border.all(color: Colors.grey, width: 4),
                    ),
                    child: _isCapturing
                        ? const CircularProgressIndicator()
                        : const Icon(
                            Icons.camera_alt,
                            color: Colors.black,
                            size: 32,
                          ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _isCapturing ? 'CAPTURING...' : 'CAPTURE REAR VIEW',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
