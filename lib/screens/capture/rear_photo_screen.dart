import 'dart:io';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import '../../models/capture_session.dart';
import '../../services/camera_permission_service.dart';
import '../../services/capture_source_service.dart';
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

  /// True while the gallery picker is open.
  ///
  /// Deliberately not [_isCapturing]: that flag drives the shutter, so sharing
  /// it made the circle spin and the label read CAPTURING... while the user
  /// was merely browsing their photographs and nothing was being captured.
  bool _isPickingFromGallery = false;

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
        ResolutionPreset.high,
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
  // CHOOSE AN EXISTING PHOTO
  // ============================================
  /// A third route into the same quality gate, beside demo and live capture.
  ///
  /// On an emulator with no camera, and on a phone where camera permission was
  /// declined, this is the only way a rear photo reaches the session at all -
  /// so it is offered in both modes and on the camera-error screen too.
  Future<void> _pickFromGallery() async {
    // The picker is a separate activity, so a second tap can land before it
    // covers the screen. This also refuses to open on top of a running capture.
    if (_isPickingFromGallery || _isCapturing || _isDisposingCamera) {
      return;
    }

    setState(() => _isPickingFromGallery = true);

    try {
      final path = await MediaPicker.pickImage();

      // Backing out of the picker is a decision, not a failure: no message,
      // no state change, no navigation.
      if (path == null) {
        return;
      }

      // MediaPicker reports a failed copy the same way it reports a
      // cancellation, so an unreadable file is the only evidence we get that
      // something went wrong after a photo really was chosen.
      if (!await _isUsableImage(path)) {
        if (mounted) {
          _showGalleryError(
            'That photo could not be read. Choose a different one, or capture it.',
          );
        }
        return;
      }

      // From here a picked photo is indistinguishable from a shot one: same
      // quality gate, same session field, same navigation on to step 4.
      if (mounted) {
        await _runQualityGate(path);
      }
    } catch (_) {
      if (mounted) {
        _showGalleryError(
          'Unable to open the gallery. Capture the photo instead.',
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isPickingFromGallery = false);
      }
    }
  }

  /// A missing or zero-byte file would reach the quality gate as a blank
  /// preview and only fail later, in scoring or the upload queue.
  Future<bool> _isUsableImage(String path) async {
    try {
      final file = File(path);
      return await file.exists() && await file.length() > 0;
    } catch (_) {
      return false;
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

  /// A SnackBar rather than the camera error dialog: a photo that will not
  /// open is worth picking again straight away, not dismissing first, and the
  /// preview underneath stays usable while it shows.
  void _showGalleryError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), duration: const Duration(seconds: 4)),
    );
  }

  /// Built once and reused by every layout below, so a picked photo cannot
  /// drift onto a different route from a captured one.
  ///
  /// Same wording, icon and styling as the other three capture steps: the
  /// worker meets this button four times in one session and it should not look
  /// like a different control each time.
  Widget _buildGalleryButton() {
    final busy = _isCapturing || _isPickingFromGallery;

    return ElevatedButton.icon(
      onPressed: busy ? null : _pickFromGallery,
      icon: _isPickingFromGallery
          ? const SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Colors.white,
              ),
            )
          : const Icon(Icons.photo_library, size: 20),
      label: Text(_isPickingFromGallery ? 'Opening...' : 'Gallery'),
      // Colours are pinned in every state because this button sits on top of a
      // photograph, where the theme's default disabled grey disappears.
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.black.withValues(alpha: 0.6),
        foregroundColor: Colors.white,
        disabledBackgroundColor: Colors.black.withValues(alpha: 0.6),
        disabledForegroundColor: Colors.white70,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      ),
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
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  _cameraError!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 18),
                ),
                const SizedBox(height: 24),
                // The camera is unavailable or was declined, so this is the
                // only way to finish step 3 - a dead end otherwise.
                _buildGalleryButton(),
              ],
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
                // Ignore the shutter while the picker is open, so the two
                // routes cannot both push a quality gate.
                onTap: _isPickingFromGallery ? null : _capturePhoto,
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

            // ============================================
            // GALLERY BUTTON GOES HERE
            // ============================================
            // Beside the shutter, not instead of it. It clears the shutter
            // column - a 72px circle plus its label reaches ~134 up from the
            // bottom - because a right-aligned labelled button any lower
            // clips the circle on a 360dp-wide phone.
            Positioned(bottom: 150, right: 20, child: _buildGalleryButton()),
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
              // Ignore the shutter while the picker is open, so the two
              // routes cannot both push a quality gate.
              onTap: _isPickingFromGallery ? null : _capturePhoto,
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

          // ============================================
          // GALLERY BUTTON GOES HERE
          // ============================================
          // Same position as the demo layout so the control does not move
          // when the mode is switched mid-demonstration.
          Positioned(bottom: 150, right: 20, child: _buildGalleryButton()),
        ],
      ),
    );
  }
}
