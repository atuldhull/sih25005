import 'package:flutter/material.dart';

import '../../models/scoring_result.dart';
import '../../widgets/result_cards.dart';
import '../assistant/chat_screen.dart';
import 'trait_detail_screen.dart';

/// Step 4.1 — Explainability Scorecard.
///
/// Renders every trait from a [ScoringResult], grouped into the five
/// model-defined categories (Dairy Strength, Rump, Feet & Legs, Udder,
/// General).
///
/// Scores are biological measurements, not quality judgments. Every score
/// and confidence bar therefore uses one neutral visual treatment — no
/// red/green, no "good"/"bad" interpretation.
///
/// Scored rows are tappable and open TraitDetailScreen (Step 4.2).
/// Not-scored rows are not tappable.
class ScorecardScreen extends StatelessWidget {
  final ScoringResult result;

  /// Optional captured photo path (e.g. side photo) passed through to
  /// TraitDetailScreen as a graceful fallback when no overlay is available.
  final String? capturedPhotoPath;

  const ScorecardScreen({
    super.key,
    required this.result,
    this.capturedPhotoPath,
  });

  /// Fixed display order for the five known categories.
  static const List<String> _categoryOrder = [
    'Dairy Strength',
    'Rump',
    'Feet & Legs',
    'Udder',
    'General',
  ];

  @override
  Widget build(BuildContext context) {
    final grouped = _groupByCategory(result.traits);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Scorecard'),
        actions: [
          IconButton(
            tooltip: 'Ask about this animal',
            icon: const Icon(Icons.chat_bubble_outline),
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (_) => ChatScreen(
                  animalId: result.animalId,
                  animalLabel: result.breedRegistered,
                ),
              ),
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Whether any of this was measured comes FIRST. Everything below it
          // is meaningless if the answer is no.
          EngineBanner(result: result),
          _HeaderCard(result: result),
          const SizedBox(height: 12),
          BreedIdentityCard(result: result),
          WeightCard(result: result),
          ScreeningCard(result: result),
          const SizedBox(height: 4),
          ..._buildCategorySections(grouped),
          if (result.farmerReport != null &&
              result.farmerReport!.trim().isNotEmpty) ...[
            const SizedBox(height: 8),
            _FarmerReportCard(text: result.farmerReport!),
          ],
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  /// Groups traits by their actual model-provided category value.
  Map<String, List<Trait>> _groupByCategory(List<Trait> traits) {
    final grouped = <String, List<Trait>>{};
    for (final trait in traits) {
      grouped.putIfAbsent(trait.category, () => []).add(trait);
    }
    return grouped;
  }

  /// Builds category sections in the fixed order, then any additional
  /// model-provided categories so every trait is always rendered.
  List<Widget> _buildCategorySections(Map<String, List<Trait>> grouped) {
    final sections = <Widget>[];
    final rendered = <String>{};

    for (final category in _categoryOrder) {
      final traits = grouped[category];
      if (traits == null || traits.isEmpty) continue;
      rendered.add(category);
      sections.addAll(_buildCategorySection(category, traits));
    }

    for (final entry in grouped.entries) {
      if (rendered.contains(entry.key)) continue;
      sections.addAll(_buildCategorySection(entry.key, entry.value));
    }

    return sections;
  }

  List<Widget> _buildCategorySection(String category, List<Trait> traits) {
    return [
      _CategoryHeader(title: category, traitCount: traits.length),
      const SizedBox(height: 8),
      for (final trait in traits) ...[
        _TraitRow(trait: trait, capturedPhotoPath: capturedPhotoPath),
        const SizedBox(height: 8),
      ],
      const SizedBox(height: 8),
    ];
  }
}

/// Compact summary of the scoring session shown above the trait list.
class _HeaderCard extends StatelessWidget {
  final ScoringResult result;

  const _HeaderCard({required this.result});

  @override
  Widget build(BuildContext context) {
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
              'Explainability Scorecard',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 12),
            _InfoRow(label: 'Animal ID', value: result.animalId),
            _InfoRow(label: 'Breed', value: result.breedRegistered),
            _InfoRow(
              label: 'Captured',
              value: _formatCapturedAt(result.capturedAt),
            ),
          ],
        ),
      ),
    );
  }

  String _formatCapturedAt(String raw) {
    final parsed = DateTime.tryParse(raw);
    if (parsed == null) return raw;
    final local = parsed.toLocal();
    final day = local.day.toString().padLeft(2, '0');
    final month = local.month.toString().padLeft(2, '0');
    final year = local.year;
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '$day/$month/$year $hour:$minute';
  }
}

/// Label/value row used in the header card.
class _InfoRow extends StatelessWidget {
  final String label;
  final String value;

  const _InfoRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 90,
            child: Text(
              label,
              style: TextStyle(fontSize: 13, color: Colors.grey.shade600),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
            ),
          ),
        ],
      ),
    );
  }
}

/// Category header with the number of traits in the category.
class _CategoryHeader extends StatelessWidget {
  final String title;
  final int traitCount;

  const _CategoryHeader({required this.title, required this.traitCount});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 8, bottom: 4),
      child: Row(
        children: [
          Expanded(
            child: Text(
              title,
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
            ),
          ),
          Text(
            '$traitCount traits',
            style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
          ),
        ],
      ),
    );
  }
}

/// A single trait row: name, score (or "Not scored"), and confidence bar.
///
/// Scored rows are tappable; not-scored rows are not.
class _TraitRow extends StatelessWidget {
  final Trait trait;
  final String? capturedPhotoPath;

  const _TraitRow({required this.trait, this.capturedPhotoPath});

  @override
  Widget build(BuildContext context) {
    final isScored = trait.score != null;

    return Card(
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: Colors.grey.shade300),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: isScored ? () => _openTraitDetail(context) : null,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: Text(
                      trait.name,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  _ScoreBadge(score: trait.score),
                ],
              ),
              const SizedBox(height: 8),
              if (!isScored) ...[
                _NotScoredNotice(reason: trait.notScoredReason),
                const SizedBox(height: 8),
              ],
              _ConfidenceBar(confidence: trait.confidence),
            ],
          ),
        ),
      ),
    );
  }

  void _openTraitDetail(BuildContext context) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => TraitDetailScreen(
          trait: trait,
          capturedPhotoPath: capturedPhotoPath,
        ),
      ),
    );
  }
}

/// Neutral score badge. A score is a biological measurement, not a quality
/// judgment, so no red/green or good/bad styling is used.
class _ScoreBadge extends StatelessWidget {
  final int? score;

  const _ScoreBadge({required this.score});

  @override
  Widget build(BuildContext context) {
    if (score == null) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: Colors.grey.shade100,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.grey.shade300),
        ),
        child: Text(
          'Not scored',
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w500,
            color: Colors.grey.shade700,
          ),
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.blueGrey.shade50,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.blueGrey.shade200),
      ),
      child: Text(
        'Score $score',
        style: TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.w600,
          color: Colors.blueGrey.shade700,
        ),
      ),
    );
  }
}

/// Neutral notice shown when the system declined to score a trait.
///
/// Refusal is an intentional system decision, not an error, so this uses
/// neutral info styling rather than red error styling.
class _NotScoredNotice extends StatelessWidget {
  final String? reason;

  const _NotScoredNotice({required this.reason});

  @override
  Widget build(BuildContext context) {
    final reasonText = reason?.trim();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline, size: 16, color: Colors.blueGrey.shade400),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'System declined to score — see reason',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    color: Colors.blueGrey,
                  ),
                ),
                if (reasonText != null && reasonText.isNotEmpty) ...[
                  const SizedBox(height: 4),
                  Text(
                    reasonText,
                    style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Small horizontal confidence bar using the actual model confidence value.
///
/// The value is clamped to 0.0–1.0 only for UI safety; the displayed
/// percentage always reflects the real model value.
class _ConfidenceBar extends StatelessWidget {
  final double? confidence;

  const _ConfidenceBar({required this.confidence});

  @override
  Widget build(BuildContext context) {
    final value = confidence;
    if (value == null) {
      return Text(
        'Confidence not available',
        style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
      );
    }

    final clamped = value.clamp(0.0, 1.0).toDouble();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              'Confidence',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
            const SizedBox(width: 8),
            Text(
              '${(value * 100).round()}%',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
            ),
          ],
        ),
        const SizedBox(height: 4),
        SizedBox(
          width: double.infinity,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: clamped,
              minHeight: 4,
              backgroundColor: Colors.grey.shade200,
              valueColor: AlwaysStoppedAnimation<Color>(
                Colors.blueGrey.shade400,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/// The plain-language summary the server writes for the farmer.
///
/// It already carries its own caveats - that the animal was not screened for
/// illness, that a placeholder is a placeholder - so it is rendered verbatim
/// rather than being re-worded here and drifting out of step with the server.
class _FarmerReportCard extends StatelessWidget {
  final String text;

  const _FarmerReportCard({required this.text});

  @override
  Widget build(BuildContext context) {
    return SectionCard(
      background: Colors.grey.shade50,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.description_outlined,
                size: 18,
                color: Colors.grey.shade700,
              ),
              const SizedBox(width: 8),
              Text(
                'Summary for the farmer',
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(text, style: const TextStyle(height: 1.45)),
        ],
      ),
    );
  }
}
