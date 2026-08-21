import 'dart:io';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import '../../models/capture_session.dart';
import '../../services/camera_permission_service.dart';
import '../../services/capture_source_service.dart';
import '../../services/demo_camera_config.dart';
import '../../services/demo_media_service.dart';
import '../../widgets/cow_silhouette_painter.dart';
import 'photo_quality_screen.dart';
import 'rear_photo_screen.dart';

class SidePhotoScreen extends StatefulWidget {
  final CaptureSession session;

  const SidePhotoScreen({super.key, required this.session});
  @override
  State<SidePhotoScreen> createState() => _SidePhotoScreenState();
}

class _SidePhotoScreenState extends State<SidePhotoScreen> {
  CameraController? _controller;
  bool _isCapturing = false;
  bool _isDisposingCamera = false;
  String? _cameraError;

  /// Demo-mode live preview asset (resolved once at startup).
  String? _demoPreviewAsset;

  /// True while the gallery picker is open.
  ///
  /// Deliberately not [_isCapturing]: the shutter must keep reading CAPTURE
  /// while someone browses their photographs, and a second tap on GALLERY
  /// before the picker has covered the screen must do nothing.
  bool _isPickingFromGallery = false;

  // ============================================
  // INITIALIZE THE BACK CAMERA
  // ============================================
  Future<void> _initializeCamera() async {
    // DEMO MODE: no real camera is initialized. The realistic side-view cow
    // image is shown behind the existing silhouette guide.
    if (DemoCameraConfig.enabled) {
      final asset = await DemoMediaService.resolveImageAsset(
        DemoMediaKind.side,
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
            _cameraError = 'Camera access is needed to capture the side photo. Allow it in Settings and try again.';
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

    // DEMO MODE: copy the bundled side-view cow image into a real temp file
    // and run the existing quality gate unchanged.
    if (DemoCameraConfig.enabled) {
      setState(() => _isCapturing = true);

      try {
        final result = await DemoMediaService.captureImage(DemoMediaKind.side);

        if (mounted) {
          await _runQualityGate(result.filePath);
        }
      } catch (_) {
        if (mounted) {
          _showError('Unable to capture the side photo.');
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
        _showError('Unable to capture the side photo.');
      }
    }

    if (mounted) {
      setState(() => _isCapturing = false);
    }
  }

  // ============================================
  // CHOOSE AN EXISTING PHOTO
  // ============================================
  /// A third route into the same quality gate, beside demo capture and the
  /// real camera. Offered in both modes because it is the only one that works
  /// on an emulator with no camera and on a phone whose camera permission was
  /// declined - and because someone holding a photograph of an animal should
  /// not have to change a setting before they can score it.
  Future<void> _pickFromGallery() async {
    // The picker is a separate activity, so a second tap can land before it
    // covers the screen. This also refuses to open on top of a running capture.
    if (_isPickingFromGallery || _isCapturing || _isDisposingCamera) {
      return;
    }

    setState(() => _isPickingFromGallery = true);

    try {
      final picked = await MediaPicker.pickImage();

      // MediaPicker returns null both when someone backs out and when the copy
      // fails, so null has to stay silent - a message after a deliberate
      // cancel would fire every time the picker is dismissed.
      if (picked != null && mounted) {
        final file = File(picked);

        // This copy is what the upload queue reads hours later, so a missing
        // or truncated one is worth catching now rather than at the end of
        // the walkthrough.
        if (await file.exists() && await file.length() > 0) {
          if (mounted) {
            // Same gate, same session field, same navigation as a shot photo:
            // downstream nothing can tell the two apart.
            await _runQualityGate(picked);
          }
        } else if (mounted) {
          _showPickerError();
        }
      }
    } catch (_) {
      if (mounted) {
        _showPickerError();
      }
    }

    if (mounted) {
      setState(() => _isPickingFromGallery = false);
    }
  }

  /// A SnackBar rather than the camera's AlertDialog: the screen behind it
  /// stays usable, so the next tap can be GALLERY again or CAPTURE.
  void _showPickerError() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        duration: Duration(seconds: 4),
        content: Text(
          'That photo could not be read. Try a different one, or use CAPTURE.',
        ),
      ),
    );
  }

  // ============================================
  // GALLERY BUTTON
  // ============================================
  /// Built once and reused by every layout below, so a picked photo cannot
  /// drift onto a different route from a captured one.
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
      // photograph, where the theme's default disabled grey disappears. The
      // disabled background stays at full opacity so the button does not seem
      // to fade away at the moment it reports that the picker is on its way.
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
  // RUN QUALITY GATE
  // ============================================
  Future<void> _runQualityGate(String path) async {
    final accepted = await Navigator.push<bool>(
      context,
      MaterialPageRoute(builder: (_) => PhotoQualityScreen(imagePath: path)),
    );

    if (!mounted) return;

    if (accepted == true) {
      widget.session.sidePhotoPath = path;

      await _disposeCameraForNavigation();

      if (!mounted) return;

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => RearPhotoScreen(session: widget.session),
        ),
      );
      return;
    }

    widget.session.sidePhotoPath = null;
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
      return 'Camera permission is required to capture the side photo.';
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
        appBar: AppBar(title: const Text('Step 2: Side View')),
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
                // Without this the screen is a dead end: no camera means no
                // shutter, and the gallery is the only way left to finish
                // step 2.
                _buildGalleryButton(),
              ],
            ),
          ),
        ),
      );
    }

    // DEMO MODE: show the realistic side-view cow image behind the existing
    // silhouette guide and capture button.
    if (DemoCameraConfig.enabled) {
      final asset = _demoPreviewAsset;
      if (asset == null) {
        return const Scaffold(body: Center(child: CircularProgressIndicator()));
      }

      return Scaffold(
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
                size: const Size(350, 500),
                painter: CowSilhouettePainter(),
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
                      _isCapturing ? 'CAPTURING...' : 'CAPTURE',
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
            // Beside the shutter, not instead of it: demo capture still
            // demonstrates the workflow, this loads a real animal. It sits
            // clear of the shutter column because a labelled button at the
            // circle's own height overlaps it on a 360dp-wide phone.
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
              size: const Size(350, 500),
              painter: CowSilhouettePainter(),
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
                    _isCapturing ? 'CAPTURING...' : 'CAPTURE',
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
