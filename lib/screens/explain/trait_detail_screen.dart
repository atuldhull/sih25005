import 'dart:io';

import 'package:flutter/material.dart';

import '../../data/trait_explanations.dart';
import '../../models/scoring_result.dart';

/// Step 4.2 — Trait detail / overlay screen.
///
/// Opened when a scored trait is tapped from [ScorecardScreen]. Shows the
/// trait's overlay image (when an overlay URL is actually available), the
/// exact measured value from the model, and a farmer-friendly explanation.
///
/// The [Trait] model does not carry an overlay URL, session ID, or photo
/// path, so this screen accepts the minimum additional information it needs
/// through constructor parameters while preserving the existing model:
///
/// * [overlayUrl] — optional backend overlay URL, e.g.
///   `/overlays/{session_id}/{trait}.jpg`. When null/empty it is never
///   constructed or fetched.
/// * [capturedPhotoPath] — optional local captured photo used as a graceful
///   fallback when no overlay is available or the overlay cannot be loaded.
///
/// This screen never crashes on missing network, missing overlay, missing
/// photo, missing measurement, or missing explanation. Every failure falls
/// back to a neutral UI state.
class TraitDetailScreen extends StatelessWidget {
  final Trait trait;
  final String? overlayUrl;
  final String? capturedPhotoPath;

  const TraitDetailScreen({
    super.key,
    required this.trait,
    this.overlayUrl,
    this.capturedPhotoPath,
  });

  @override
  Widget build(BuildContext context) {
    final isScored = trait.score != null;

    return Scaffold(
      appBar: AppBar(title: Text(trait.name)),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _ImageSection(
            trait: trait,
            overlayUrl: overlayUrl,
            capturedPhotoPath: capturedPhotoPath,
          ),
          const SizedBox(height: 16),
          if (isScored) ...[
            _MeasuredValueCard(trait: trait),
            const SizedBox(height: 16),
            _ExplanationCard(trait: trait),
            const SizedBox(height: 16),
            _ScoreCard(trait: trait),
          ] else ...[
            _NotScoredCard(trait: trait),
          ],
        ],
      ),
    );
  }
}

/// Image/overlay section.
///
/// Priority:
/// 1. Overlay URL when actually available and loadable.
/// 2. Captured photo when the overlay is unavailable/offline/failed.
/// 3. Neutral placeholder when neither is available.
class _ImageSection extends StatelessWidget {
  final Trait trait;
  final String? overlayUrl;
  final String? capturedPhotoPath;

  const _ImageSection({
    required this.trait,
    this.overlayUrl,
    this.capturedPhotoPath,
  });

  @override
  Widget build(BuildContext context) {
    final overlay = overlayUrl?.trim();
    final photo = capturedPhotoPath?.trim();

    // Overlay is only attempted when a real URL is provided.
    if (overlay != null && overlay.isNotEmpty) {
      return _OverlayImage(
        overlayUrl: overlay,
        capturedPhotoPath: (photo != null && photo.isNotEmpty) ? photo : null,
      );
    }

    // No overlay URL — fall back to the captured photo if one exists.
    if (photo != null && photo.isNotEmpty) {
      return _CapturedPhoto(path: photo, note: 'Overlay pending sync');
    }

    // Neither overlay nor photo available.
    return const _UnavailablePlaceholder();
  }
}

/// Attempts to load the overlay image, falling back to the captured photo
/// (or a neutral placeholder) on any failure.
class _OverlayImage extends StatelessWidget {
  final String overlayUrl;
  final String? capturedPhotoPath;

  const _OverlayImage({required this.overlayUrl, this.capturedPhotoPath});

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Image.network(
        overlayUrl,
        fit: BoxFit.cover,
        width: double.infinity,
        height: 260,
        loadingBuilder: (context, child, progress) {
          if (progress == null) return child;
          return _LoadingBox();
        },
        errorBuilder: (context, error, stackTrace) {
          // Overlay failed to load — fall back to the captured photo.
          final photo = capturedPhotoPath;
          if (photo != null && photo.isNotEmpty) {
            return _CapturedPhoto(
              path: photo,
              note: 'Overlay unavailable — showing captured photo',
            );
          }
          return const _UnavailablePlaceholder();
        },
      ),
    );
  }
}

/// Displays a local captured photo file with a small neutral note.
class _CapturedPhoto extends StatelessWidget {
  final String path;
  final String note;

  const _CapturedPhoto({required this.path, required this.note});

  @override
  Widget build(BuildContext context) {
    final file = File(path);
    final exists = file.existsSync();

    if (!exists) {
      return const _UnavailablePlaceholder();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Image.file(
            file,
            fit: BoxFit.cover,
            width: double.infinity,
            height: 260,
            errorBuilder: (context, error, stackTrace) {
              return const _UnavailablePlaceholder();
            },
          ),
        ),
        const SizedBox(height: 8),
        Text(note, style: TextStyle(fontSize: 12, color: Colors.grey.shade600)),
      ],
    );
  }
}

/// Neutral placeholder shown when no overlay and no photo are available.
class _UnavailablePlaceholder extends StatelessWidget {
  const _UnavailablePlaceholder();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      height: 260,
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.image_not_supported_outlined,
            size: 40,
            color: Colors.grey.shade400,
          ),
          const SizedBox(height: 8),
          Text(
            'Overlay unavailable',
            style: TextStyle(fontSize: 13, color: Colors.grey.shade600),
          ),
        ],
      ),
    );
  }
}

/// Neutral loading box shown while the overlay image downloads.
class _LoadingBox extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      height: 260,
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: const Center(child: CircularProgressIndicator()),
    );
  }
}

/// Displays the exact measured value from the model.
class _MeasuredValueCard extends StatelessWidget {
  final Trait trait;

  const _MeasuredValueCard({required this.trait});

  @override
  Widget build(BuildContext context) {
    final value = trait.valueWithInterval?.trim();

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: Colors.grey.shade300),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Measured value',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
            const SizedBox(height: 4),
            Text(
              (value != null && value.isNotEmpty)
                  ? value
                  : 'Measured value not available',
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
            ),
          ],
        ),
      ),
    );
  }
}

/// Farmer-friendly explanation for the trait.
class _ExplanationCard extends StatelessWidget {
  final Trait trait;

  const _ExplanationCard({required this.trait});

  @override
  Widget build(BuildContext context) {
    final explanation = traitExplanations[trait.name];

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: Colors.grey.shade300),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'What this measures',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
            const SizedBox(height: 4),
            Text(
              explanation ??
                  'A detailed explanation for this trait is not available yet.',
              style: const TextStyle(fontSize: 14, height: 1.4),
            ),
          ],
        ),
      ),
    );
  }
}

/// Neutral score and confidence display. A score is a biological measurement,
/// not a quality judgment, so no red/green or good/bad styling is used.
class _ScoreCard extends StatelessWidget {
  final Trait trait;

  const _ScoreCard({required this.trait});

  @override
  Widget build(BuildContext context) {
    final score = trait.score;
    final confidence = trait.confidence;

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: Colors.grey.shade300),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Score',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
            const SizedBox(height: 4),
            Text(
              score != null ? '$score' : 'Not scored',
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
            ),
            if (confidence != null) ...[
              const SizedBox(height: 12),
              Text(
                'Confidence ${(confidence * 100).round()}%',
                style: TextStyle(fontSize: 13, color: Colors.grey.shade700),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Defensive state for a not-scored trait passed to this screen.
///
/// Refusal is an intentional system decision, not an error, so this uses
/// neutral info styling rather than red warning styling.
class _NotScoredCard extends StatelessWidget {
  final Trait trait;

  const _NotScoredCard({required this.trait});

  @override
  Widget build(BuildContext context) {
    final reason = trait.notScoredReason?.trim();

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: Colors.grey.shade300),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.info_outline,
                  size: 16,
                  color: Colors.blueGrey.shade400,
                ),
                const SizedBox(width: 8),
                const Text(
                  'Not scored',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
              ],
            ),
            if (reason != null && reason.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                reason,
                style: TextStyle(fontSize: 13, color: Colors.grey.shade700),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
