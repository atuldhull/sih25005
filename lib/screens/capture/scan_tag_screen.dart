import 'package:flutter/material.dart';

class ScanTagScreen extends StatefulWidget {
  const ScanTagScreen({super.key});

  @override
  State<ScanTagScreen> createState() => _ScanTagScreenState();
}

class _ScanTagScreenState extends State<ScanTagScreen> {
  bool _isInitializing = true;
  bool _isCapturing = false;

  // Demo image state
  bool _isCaptured = false;

  final TextEditingController _tagController = TextEditingController();

  @override
  void initState() {
    super.initState();

    // Simulate loading the camera/demo capture screen.
    Future.delayed(const Duration(milliseconds: 500), () {
      if (!mounted) return;

      setState(() {
        _isInitializing = false;
      });
    });
  }

  // ------------------------------------------------------------
  // CAPTURE DEMO IMAGE
  // ------------------------------------------------------------

  Future<void> _captureTagImage() async {
    if (_isCapturing || _isCaptured) {
      return;
    }

    setState(() {
      _isCapturing = true;
    });

    // Simulate a real camera capture.
    await Future.delayed(const Duration(milliseconds: 700));

    if (!mounted) return;

    setState(() {
      _isCaptured = true;
      _isCapturing = false;
    });
  }

  // ------------------------------------------------------------
  // RETAKE IMAGE
  // ------------------------------------------------------------

  void _retakeImage() {
    setState(() {
      _isCaptured = false;
    });
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
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Ear tag $animalId confirmed successfully.')),
    );

    // Step 1.8 will connect the next capture screen here.
  }

  // ------------------------------------------------------------
  // DISPOSE
  // ------------------------------------------------------------

  @override
  void dispose() {
    _tagController.dispose();
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
                Expanded(
                  child: _isCaptured
                      ? _buildCapturedImagePreview()
                      : _buildDemoCowPreview(),
                ),

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
  // DEMO COW IMAGE
  // ------------------------------------------------------------

  Widget _buildDemoCowPreview() {
    return Stack(
      fit: StackFit.expand,
      children: [
        Image.asset('assets/demo/cow_ear_tag.png', fit: BoxFit.cover),

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
    return Stack(
      fit: StackFit.expand,
      children: [
        Image.asset('assets/demo/cow_ear_tag.png', fit: BoxFit.cover),

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
    return SizedBox(
      width: double.infinity,
      height: 55,
      child: ElevatedButton.icon(
        onPressed: _isCapturing ? null : _captureTagImage,
        icon: const Icon(Icons.camera_alt),
        label: Text(_isCapturing ? 'Capturing...' : 'Capture Ear Tag'),
      ),
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
