import 'dart:io';

import 'package:flutter/material.dart';

class PhotoQualityScreen extends StatefulWidget {
  final String imagePath;

  const PhotoQualityScreen({super.key, required this.imagePath});

  @override
  State<PhotoQualityScreen> createState() => _PhotoQualityScreenState();
}

class _PhotoQualityScreenState extends State<PhotoQualityScreen> {
  bool _canConfirm = false;
  bool _sharpnessAccepted = false;

  @override
  void initState() {
    super.initState();

    // Give the worker a moment to inspect the captured image.
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) {
        setState(() {
          _canConfirm = true;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Fullscreen image preview
          Positioned.fill(
            child: Image.file(File(widget.imagePath), fit: BoxFit.contain),
          ),

          // Confirmation UI
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: const EdgeInsets.all(20).copyWith(bottom: 40),
              color: Colors.black.withValues(alpha: 0.7),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    _sharpnessAccepted
                        ? 'Is the whole animal visible in frame?'
                        : 'Is this photo sharp?',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 20),
                  Row(
                    children: [
                      // RETAKE button
                      Expanded(
                        child: OutlinedButton(
                          onPressed: () => Navigator.pop(context, false),
                          style: OutlinedButton.styleFrom(
                            foregroundColor: Colors.white,
                            side: const BorderSide(color: Colors.white),
                            minimumSize: const Size(double.infinity, 55),
                          ),
                          child: const Text('RETAKE'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      // YES button
                      Expanded(
                        child: ElevatedButton(
                          onPressed: _canConfirm
                              ? () {
                                  if (_sharpnessAccepted) {
                                    Navigator.pop(context, true);
                                    return;
                                  }

                                  setState(() {
                                    _sharpnessAccepted = true;
                                  });
                                }
                              : null,
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size(double.infinity, 55),
                          ),
                          child: const Text('YES'),
                        ),
                      ),
                    ],
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
