import 'package:flutter/material.dart';

import '../models/scoring_result.dart';

/// The cards that sit above the trait list: which engine answered, what the
/// breed model saw, what the animal weighs, and whether anything looked at
/// her health.
///
/// They live together in one file because they share a rule: every one of them
/// exists to stop a number being read as more certain than it is. Splitting
/// them up tends to mean one gets updated and the others quietly do not.

/// Rounded outline card, matching the scorecard's existing treatment.
class SectionCard extends StatelessWidget {
  final Widget child;
  final Color? background;
  final Color? border;
  final EdgeInsetsGeometry padding;

  const SectionCard({
    super.key,
    required this.child,
    this.background,
    this.border,
    this.padding = const EdgeInsets.all(16),
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      color: background,
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: border ?? Colors.grey.shade300),
      ),
      child: Padding(padding: padding, child: child),
    );
  }
}

/// Says, in plain words, whether these numbers were measured.
///
/// The baseline engine invents all twenty scores, a weight, and symptoms. It
/// used to be disclosed as a small grey "Baseline" label beside twenty
/// confident-looking figures, which nobody reads as "none of this is real".
class EngineBanner extends StatelessWidget {
  final ScoringResult result;

  const EngineBanner({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    if (result.isDemonstration) {
      return SectionCard(
        background: const Color(0xFFFDECEA),
        border: const Color(0xFFD32F2F),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.warning_amber_rounded, color: Color(0xFFD32F2F)),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'DEMONSTRATION DATA',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: const Color(0xFFB71C1C),
                      letterSpacing: 0.5,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'These scores were NOT measured from the photographs. '
                    'Nothing here describes this animal. Do not act on the '
                    'weight, the scores, or the health notes.',
                    style: TextStyle(color: Color(0xFFB71C1C), height: 1.35),
                  ),
                ],
              ),
            ),
          ],
        ),
      );
    }

    final scored = result.scoredCount;
    final total = result.traits.length;
    return SectionCard(
      background: const Color(0xFFE8F5E9),
      border: const Color(0xFF2E7D32),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.verified_outlined, color: Color(0xFF2E7D32)),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'MEASURED',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: const Color(0xFF1B5E20),
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '$scored of $total traits were measured from these '
                  'photographs. The other ${total - scored} were refused — '
                  'tap any of them to read why.',
                  style: const TextStyle(
                    color: Color(0xFF1B5E20),
                    height: 1.35,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// What the breed models actually claim, and what they refuse to claim.
class BreedIdentityCard extends StatelessWidget {
  final ScoringResult result;

  const BreedIdentityCard({super.key, required this.result});

  static String _pretty(String raw) =>
      raw.replaceAll('_', ' ').trim().toLowerCase();

  @override
  Widget build(BuildContext context) {
    final registered = result.breedRegistered;
    final group = result.predictedGroup;
    final species = result.predictedSpecies;

    return SectionCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _CardTitle('Breed identity', icon: Icons.pets_outlined),
          const SizedBox(height: 12),
          _Line(label: 'Registered as', value: registered),

          if (species != null && species.isNotEmpty)
            _Line(
              label: 'Species seen',
              value: _withConfidence(_pretty(species), result.speciesConfidence),
              note: result.speciesConsistent == false
                  ? 'does NOT match the record — worth checking'
                  : null,
              warn: result.speciesConsistent == false,
            ),

          if (group != null && group.isNotEmpty)
            _Line(
              label: 'Breed group',
              value: _withConfidence(_pretty(group), result.groupConfidence),
              note: _groupNote(),
              warn: result.groupConsistent == false,
            ),

          const SizedBox(height: 10),
          const Divider(height: 1),
          const SizedBox(height: 10),
          Text(
            _verifyWording(registered),
            style: TextStyle(
              color: result.breedVerifyStatus == 'disagree'
                  ? const Color(0xFFB71C1C)
                  : Colors.grey.shade700,
              height: 1.35,
              fontSize: 13,
            ),
          ),
        ],
      ),
    );
  }

  String _withConfidence(String label, double? confidence) {
    if (confidence == null) return label;
    return '$label — ${(confidence * 100).round()}% confident';
  }

  String? _groupNote() {
    if (result.groupReliable == false) {
      return 'below this group’s own reliability bar — treat as a hint';
    }
    if (result.groupConsistent == false) {
      return 'does NOT match ${result.breedRegistered} — worth checking the '
          'record';
    }
    if (result.groupConsistent == true) {
      return 'consistent with ${result.breedRegistered}';
    }
    return null;
  }

  /// `breed_verified == false` means NOT CHECKED. Rendering it as a cross
  /// would accuse every correctly registered animal in the district.
  String _verifyWording(String registered) {
    switch (result.breedVerifyStatus) {
      case 'agree':
        return 'The photograph matches the registered breed.';
      case 'disagree':
        return 'The photograph does NOT match $registered. This is worth a '
            'human check of the record — it is not an automatic correction.';
      default:
        return 'The exact-breed model is switched off: it measured 38.1% on '
            'photographs from a source it had never seen, which is not worth '
            'acting on. This is NOT a mismatch — the exact breed was never '
            'checked.';
    }
  }
}

/// The weight, its method, and the second opinion that may contradict it.
class WeightCard extends StatelessWidget {
  final ScoringResult result;

  const WeightCard({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final range = result.weightRange;

    if (range == null) {
      return SectionCard(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _CardTitle('Weight', icon: Icons.monitor_weight_outlined),
            const SizedBox(height: 10),
            Text(
              'Not estimated.',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            Text(
              result.weightKg['reason']?.toString() ??
                  'A weight needs an object of known size in the frame. '
                      'Photograph the ear tag close up and this fills in.',
              style: TextStyle(color: Colors.grey.shade700, height: 1.35),
            ),
          ],
        ),
      );
    }

    final crossCheck = result.weightCrossCheck;
    final disagrees =
        crossCheck != null && crossCheck.toUpperCase().contains('DISAGREE');

    return SectionCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _CardTitle('Weight', icon: Icons.monitor_weight_outlined),
          const SizedBox(height: 10),
          Text(
            range,
            style: Theme.of(
              context,
            ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 4),
          Text(
            'a range, not a single figure — the estimate is only as good as '
            'the scale it was built on',
            style: TextStyle(color: Colors.grey.shade600, fontSize: 12.5),
          ),
          if (result.weightMethod != null) ...[
            const SizedBox(height: 10),
            _Line(label: 'Method', value: result.weightMethod!),
          ],
          if (crossCheck != null) ...[
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: disagrees
                    ? const Color(0xFFFFF3E0)
                    : Colors.grey.shade100,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: disagrees
                      ? const Color(0xFFEF6C00)
                      : Colors.grey.shade300,
                ),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    disagrees ? Icons.info_outline : Icons.check_circle_outline,
                    size: 18,
                    color: disagrees
                        ? const Color(0xFFEF6C00)
                        : Colors.grey.shade600,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Second, independent estimate',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: Colors.grey.shade800,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          crossCheck,
                          style: const TextStyle(fontSize: 13, height: 1.3),
                        ),
                        if (disagrees) ...[
                          const SizedBox(height: 6),
                          const Text(
                            'The two methods do not agree, so treat the weight '
                            'as indicative. Do not sell on this number.',
                            style: TextStyle(
                              fontSize: 12.5,
                              height: 1.3,
                              color: Color(0xFFE65100),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }
}

/// Whether anything examined this animal for illness.
///
/// An empty symptom list was previously reported to the farmer as "No health
/// problems were flagged", which reads as a clinical finding on an animal
/// nothing had looked at.
class ScreeningCard extends StatelessWidget {
  final ScoringResult result;

  const ScreeningCard({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final risks = result.riskReport;
    final screened = result.vetScreened;

    if (!screened) {
      return SectionCard(
        background: Colors.grey.shade100,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _CardTitle('Health screening', icon: Icons.healing_outlined),
            const SizedBox(height: 8),
            Text(
              'NOT SCREENED',
              style: Theme.of(context).textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.bold,
                letterSpacing: 0.5,
                color: Colors.grey.shade800,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'This build has no trained symptom detector, so nothing examined '
              'this animal for illness. That is not a clean bill of health — '
              'if she seems unwell, see a veterinarian.',
              style: TextStyle(color: Colors.grey.shade700, height: 1.35),
            ),
          ],
        ),
      );
    }

    return SectionCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _CardTitle('Health screening', icon: Icons.healing_outlined),
          const SizedBox(height: 8),
          if (risks.isEmpty)
            Text(
              'Screened, nothing flagged. A screening is not a diagnosis.',
              style: TextStyle(color: Colors.grey.shade700, height: 1.35),
            )
          else
            ...risks.whereType<Map>().map((r) {
              final name = r['condition']?.toString() ?? 'Finding';
              final band = r['risk_band']?.toString() ?? '';
              final why = r['reason']?.toString();
              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      band.isEmpty ? name : '$name — $band risk',
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    if (why != null && why.isNotEmpty)
                      Text(
                        why,
                        style: TextStyle(
                          color: Colors.grey.shade700,
                          fontSize: 13,
                          height: 1.3,
                        ),
                      ),
                  ],
                ),
              );
            }),
          const SizedBox(height: 6),
          Text(
            'A screening from photographs and video, not a diagnosis. '
            'A veterinarian makes the final call.',
            style: TextStyle(
              color: Colors.grey.shade600,
              fontSize: 12.5,
              height: 1.3,
            ),
          ),
        ],
      ),
    );
  }
}

// --- small shared pieces --------------------------------------------------

class _CardTitle extends StatelessWidget {
  final String text;
  final IconData icon;

  const _CardTitle(this.text, {required this.icon});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, size: 18, color: Colors.grey.shade700),
        const SizedBox(width: 8),
        Text(
          text,
          style: Theme.of(
            context,
          ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
        ),
      ],
    );
  }
}

class _Line extends StatelessWidget {
  final String label;
  final String value;
  final String? note;
  final bool warn;

  const _Line({
    required this.label,
    required this.value,
    this.note,
    this.warn = false,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(
                width: 110,
                child: Text(
                  label,
                  style: TextStyle(color: Colors.grey.shade600, fontSize: 13),
                ),
              ),
              Expanded(
                child: Text(
                  value,
                  style: const TextStyle(fontSize: 14.5, height: 1.3),
                ),
              ),
            ],
          ),
          if (note != null)
            Padding(
              padding: const EdgeInsets.only(left: 110, top: 2),
              child: Text(
                note!,
                style: TextStyle(
                  fontSize: 12.5,
                  height: 1.3,
                  color: warn ? const Color(0xFFB71C1C) : Colors.grey.shade600,
                ),
              ),
            ),
        ],
      ),
    );
  }
}
