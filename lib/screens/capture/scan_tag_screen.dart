import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../../models/capture_session.dart';
import '../../services/camera_permission_service.dart';
import '../../services/capture_source_service.dart';
import '../../services/demo_camera_config.dart';
import '../../services/demo_media_service.dart';
import 'side_photo_screen.dart';

class ScanTagScreen extends StatefulWidget {
  const ScanTagScreen({super.key});

  @override
  State<ScanTagScreen> createState() => _ScanTagScreenState();
}

class _ScanTagScreenState extends State<ScanTagScreen> {
  bool _isInitializing = true;
  bool _isCapturing = false;

  /// Tracked separately from [_isCapturing] so the shutter button does not
  /// read "Capturing..." while the user is still browsing the gallery.
  bool _isPickingFromGallery = false;

  bool _isCaptured = false;
  CameraController? _controller;
  String? _capturedTagImagePath;
  String? _cameraError;

  /// Demo-mode live preview asset (resolved once at startup).
  String? _demoPreviewAsset;

  final TextEditingController _tagController = TextEditingController();

  /// One in-flight image request at a time. Both buttons read this, so a
  /// second tap - on either of them - cannot start a competing capture.
  bool get _isBusy => _isCapturing || _isPickingFromGallery;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  // ------------------------------------------------------------
  // INITIALIZE CAMERA
  // ------------------------------------------------------------

  Future<void> _initializeCamera() async {
    // DEMO MODE: no real camera is initialized. The realistic cow image is
    // shown behind the existing ear-tag guide.
    if (DemoCameraConfig.enabled) {
      final asset = await DemoMediaService.resolveImageAsset(
        DemoMediaKind.earTag,
      );

      if (!mounted) return;

      setState(() {
        _demoPreviewAsset = asset;
        _cameraError = null;
        _isInitializing = false;
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
            _cameraError = 'Camera access is needed to capture the ear tag. Allow it in Settings and try again.';
            _isInitializing = false;
          });
        }
        return;
      }

      final cameras = await availableCameras();

      if (cameras.isEmpty) {
        if (mounted) {
          setState(() {
            _cameraError = 'No camera was found on this device.';
            _isInitializing = false;
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
        _isInitializing = false;
      });
    } on CameraException catch (e) {
      if (mounted) {
        setState(() {
          _cameraError = _cameraMessage(e);
          _isInitializing = false;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _cameraError = 'Unable to initialize the camera.';
          _isInitializing = false;
        });
      }
    }
  }

  // ------------------------------------------------------------
  // CAPTURE EAR TAG IMAGE
  // ------------------------------------------------------------

  Future<void> _captureTagImage() async {
    if (_isCapturing || _isCaptured) {
      return;
    }

    // DEMO MODE: copy the bundled cow/tag image into a real temp file so the
    // existing captured-preview and retake-cleanup flow works unchanged.
    if (DemoCameraConfig.enabled) {
      setState(() {
        _isCapturing = true;
      });

      try {
        final result = await DemoMediaService.captureImage(
          DemoMediaKind.earTag,
        );

        if (!mounted) return;

        setState(() {
          _capturedTagImagePath = result.filePath;
          _isCaptured = true;
          _isCapturing = false;
        });
      } catch (_) {
        if (mounted) {
          setState(() {
            _isCapturing = false;
          });

          _showCameraError('Unable to capture the ear tag image.');
        }
      }
      return;
    }

    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      _showCameraError('Camera is not ready yet.');
      return;
    }

    setState(() {
      _isCapturing = true;
    });

    try {
      final photo = await controller.takePicture();

      if (!mounted) return;

      setState(() {
        _capturedTagImagePath = photo.path;
        _isCaptured = true;
        _isCapturing = false;
      });
    } on CameraException catch (e) {
      if (mounted) {
        setState(() {
          _isCapturing = false;
        });

        _showCameraError(_cameraMessage(e));
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _isCapturing = false;
        });

        _showCameraError('Unable to capture the ear tag image.');
      }
    }
  }

  // ------------------------------------------------------------
  // PICK EAR TAG IMAGE FROM GALLERY
  // ------------------------------------------------------------

  /// Third route to an ear-tag photograph, beside demo media and the real
  /// shutter. It is offered in BOTH modes because it is the only one that
  /// survives an emulator with no camera and a phone whose camera permission
  /// was declined - and because a judge may simply hand over a phone that
  /// already has a photograph of the animal on it.
  ///
  /// It deliberately ends in the same state as a capture - same path variable,
  /// same _isCaptured flag - so the "Check the ear tag image" / Retake /
  /// Confirm step, and everything downstream that derives centimetres from the
  /// tag, cannot tell a picked photograph from a shot one.
  Future<void> _pickTagImageFromGallery() async {
    if (_isBusy || _isCaptured) {
      return;
    }

    setState(() {
      _isPickingFromGallery = true;
    });

    try {
      // MediaPicker copies the chosen file into this app's cache, so the path
      // it returns is ours to keep - and safe for _retakeImage to delete
      // without touching the original in the user's gallery.
      final path = await MediaPicker.pickImage();

      if (!mounted) return;

      if (path == null) {
        // Backed out of the picker, or the pick failed and was swallowed.
        // Either way nothing was chosen, so nothing here changes.
        setState(() {
          _isPickingFromGallery = false;
        });
        return;
      }

      // A zero-byte copy means the source could not be read - a cloud-only
      // photo that never downloaded, most often. Better caught now than as a
      // blank tag at the measurement step.
      final file = File(path);
      if (!await file.exists() || await file.length() == 0) {
        if (!mounted) return;

        setState(() {
          _isPickingFromGallery = false;
        });

        _showPickerError(
          'That photo could not be read. Try another, or download it to the device first.',
        );
        return;
      }

      if (!mounted) return;

      setState(() {
        _capturedTagImagePath = path;
        _isCaptured = true;
        _isPickingFromGallery = false;
      });
    } catch (_) {
      if (mounted) {
        setState(() {
          _isPickingFromGallery = false;
        });

        _showPickerError('Unable to open the gallery. You can capture the ear tag instead.');
      }
    }
  }

  // ------------------------------------------------------------
  // RETAKE IMAGE
  // ------------------------------------------------------------

  Future<void> _retakeImage() async {
    final oldPath = _capturedTagImagePath;

    setState(() {
      _isCaptured = false;
      _capturedTagImagePath = null;
    });

    if (oldPath != null) {
      await _deleteTemporaryFile(oldPath);
    }
  }

  // ------------------------------------------------------------
  // CONFIRM IMAGE
  // ------------------------------------------------------------

  void _confirmImage() {
    if (!_isCaptured) {
      return;
    }

    _showTagIdDialog();
  }

  // ------------------------------------------------------------
  // ENTER TAG ID
  // ------------------------------------------------------------

  void _showTagIdDialog() {
    _tagController.clear();

    showDialog(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          title: const Text('Enter Ear Tag ID'),
          content: TextField(
            controller: _tagController,
            keyboardType: TextInputType.number,
            maxLength: 12,
            autofocus: true,
            decoration: const InputDecoration(
              hintText: 'Enter 12-digit tag number',
              border: OutlineInputBorder(),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(dialogContext);
              },
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                final String tag = _tagController.text.trim();

                if (tag.length != 12 || int.tryParse(tag) == null) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Tag ID must contain exactly 12 digits.'),
                    ),
                  );
                  return;
                }

                Navigator.pop(dialogContext);

                _goToNextStep(tag);
              },
              child: const Text('Confirm'),
            ),
          ],
        );
      },
    );
  }

  // ------------------------------------------------------------
  // NEXT STEP
  // ------------------------------------------------------------

  void _goToNextStep(String animalId) {
    // _capturedTagImagePath was already being taken here and then dropped,
    // because CaptureSession had nowhere to put it. It is the only object of
    // known size in any of the photographs, so it is what makes a centimetre
    // scale possible at all.
    final session = CaptureSession(
      tagId: animalId,
      tagPhotoPath: _capturedTagImagePath,
    );

    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => SidePhotoScreen(session: session)),
    );
  }

  // ------------------------------------------------------------
  // DISPOSE
  // ------------------------------------------------------------

  @override
  void dispose() {
    _tagController.dispose();
    _controller?.dispose();
    super.dispose();
  }

  // ------------------------------------------------------------
  // BUILD
  // ------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Scan Ear Tag')),
      body: _isInitializing
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Expanded(child: _buildPreview()),

                Padding(
                  padding: const EdgeInsets.all(20),
                  child: _isCaptured
                      ? _buildConfirmationButtons()
                      : _buildCaptureButton(),
                ),
              ],
            ),
    );
  }

  // ------------------------------------------------------------
  // CAMERA / CAPTURED PREVIEW
  // ------------------------------------------------------------

  Widget _buildPreview() {
    // Checked ahead of _cameraError: a gallery pick works even when the camera
    // never initialized, and on that phone the error text would otherwise sit
    // over the picked photo and hide the image the user is being asked to
    // check before confirming. A capture cannot reach here with an error set,
    // so the camera path is unaffected.
    if (_isCaptured) {
      return _buildCapturedImagePreview();
    }

    if (_cameraError != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            _cameraError!,
            textAlign: TextAlign.center,
            style: const TextStyle(fontSize: 18),
          ),
        ),
      );
    }

    // DEMO MODE: realistic cow image behind the existing ear-tag guide.
    if (DemoCameraConfig.enabled) {
      final asset = _demoPreviewAsset;
      if (asset == null) {
        return const Center(child: CircularProgressIndicator());
      }

      return Stack(
        fit: StackFit.expand,
        children: [
          Image.asset(asset, fit: BoxFit.cover),

          // Slight dark overlay
          Container(color: Colors.black.withValues(alpha: 0.15)),

          // Ear-tag guide box
          Center(
            child: Container(
              width: 280,
              height: 180,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.white, width: 3),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Center(
                child: Text(
                  'Place ear tag here',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    shadows: [Shadow(color: Colors.black, blurRadius: 6)],
                  ),
                ),
              ),
            ),
          ),
        ],
      );
    }

    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(controller),

        // Slight dark overlay
        Container(color: Colors.black.withValues(alpha: 0.15)),

        // Ear-tag guide box
        Center(
          child: Container(
            width: 280,
            height: 180,
            decoration: BoxDecoration(
              border: Border.all(color: Colors.white, width: 3),
              borderRadius: BorderRadius.circular(16),
            ),
            child: const Center(
              child: Text(
                'Place ear tag here',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  shadows: [Shadow(color: Colors.black, blurRadius: 6)],
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  // ------------------------------------------------------------
  // CAPTURED IMAGE PREVIEW
  // ------------------------------------------------------------

  Widget _buildCapturedImagePreview() {
    final path = _capturedTagImagePath;

    return Stack(
      fit: StackFit.expand,
      children: [
        if (path != null)
          Image.file(File(path), fit: BoxFit.cover)
        else
          const ColoredBox(color: Colors.black),

        Container(color: Colors.black.withValues(alpha: 0.12)),

        // Top label
        Positioned(
          top: 20,
          left: 20,
          right: 20,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.65),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Text(
              'Check the ear tag image',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),

        // Captured indicator
        Positioned(
          top: 80,
          right: 20,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.green,
              borderRadius: BorderRadius.circular(20),
            ),
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.check_circle, color: Colors.white, size: 20),
                SizedBox(width: 6),
                Text(
                  'Captured',
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  // ------------------------------------------------------------
  // CAPTURE BUTTON
  // ------------------------------------------------------------

  Widget _buildCaptureButton() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: double.infinity,
          height: 55,
          child: ElevatedButton.icon(
            onPressed: _isBusy ? null : _captureTagImage,
            icon: const Icon(Icons.camera_alt),
            label: Text(_isCapturing ? 'Capturing...' : 'Capture Ear Tag'),
          ),
        ),

        const SizedBox(height: 12),

        // Full width under the shutter rather than squeezed alongside it: at
        // half width the labels wrap, and this button is the fallback that has
        // to stay readable on the phone where the camera one did not work.
        SizedBox(
          width: double.infinity,
          height: 55,
          child: OutlinedButton.icon(
            onPressed: _isBusy ? null : _pickTagImageFromGallery,
            // Same icon and wording as steps 2, 3 and 4. The button is shaped
            // differently here because this screen has a real button bar
            // rather than a photograph to float over, but it must still read
            // as the same action four times in one session.
            icon: const Icon(Icons.photo_library),
            label: Text(_isPickingFromGallery ? 'Opening...' : 'Gallery'),
            style: OutlinedButton.styleFrom(
              minimumSize: const Size(double.infinity, 55),
            ),
          ),
        ),
      ],
    );
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
      return 'Camera permission is required to capture the ear tag.';
    }

    return 'Unable to use the camera. Please check permissions and try again.';
  }

  /// A failed pick is not a camera fault and must not look like one: it leaves
  /// the screen usable with both routes still open, so a SnackBar rather than
  /// the blocking dialog [_showCameraError] puts up.
  void _showPickerError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  void _showCameraError(String message) {
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

  // ------------------------------------------------------------
  // RETAKE + CONFIRM BUTTONS
  // ------------------------------------------------------------

  Widget _buildConfirmationButtons() {
    return Row(
      children: [
        Expanded(
          child: OutlinedButton.icon(
            onPressed: _retakeImage,
            icon: const Icon(Icons.refresh),
            label: const Text('Retake'),
            style: OutlinedButton.styleFrom(
              minimumSize: const Size(double.infinity, 55),
            ),
          ),
        ),

        const SizedBox(width: 12),

        Expanded(
          child: ElevatedButton.icon(
            onPressed: _confirmImage,
            icon: const Icon(Icons.check),
            label: const Text('Confirm'),
            style: ElevatedButton.styleFrom(
              minimumSize: const Size(double.infinity, 55),
            ),
          ),
        ),
      ],
    );
  }
}
