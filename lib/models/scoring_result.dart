/// One conformation trait as the server reports it.
///
/// A trait with `score == null` is REFUSED, not missing: the pipeline had a
/// look and declined to answer. [notScoredReason] carries why, and it is the
/// most useful text on the whole scorecard - showing the row without it turns
/// an honest refusal into a blank.
class Trait {
  final String name;
  final String category;
  final int? score;
  final double? confidence;
  final String? measuredValue;
  final String measureClass;
  final List<List<int>> overlayPoints;
  final String? notScoredReason;

  /// Confidence interval on the measurement, already formatted by the server,
  /// e.g. `146.8-182.2 cm`. Null when the trait did not measure.
  ///
  /// A string rather than a pair of numbers because that is what the server
  /// sends - it owns the unit and the rounding, and re-deriving them here
  /// would mean two places to keep in step. A single centimetre figure without
  /// this reads as far more certain than the pipeline ever claimed.
  final String? ci;

  /// Which photograph the trait was measured from - `side` or `rear`.
  final String? view;

  Trait({
    required this.name,
    required this.category,
    this.score,
    this.confidence,
    this.measuredValue,
    required this.measureClass,
    this.overlayPoints = const [],
    this.notScoredReason,
    this.ci,
    this.view,
  });

  /// True when the pipeline produced a number for this trait.
  bool get isScored => score != null;

  /// The measured value with its interval, ready to render, or null.
  ///
  /// e.g. `164.5 cm  (146.8-182.2 cm)`
  String? get valueWithInterval {
    final v = measuredValue;
    if (v == null || v.isEmpty) return null;
    final bounds = ci;
    if (bounds == null || bounds.isEmpty) return v;
    return '$v  ($bounds)';
  }

  factory Trait.fromJson(Map<String, dynamic> j) => Trait(
    name: (j['name'] ?? 'Unknown trait').toString(),
    category: (j['category'] ?? 'General').toString(),
    score: (j['score'] as num?)?.toInt(),
    confidence: (j['confidence'] as num?)?.toDouble(),
    measuredValue: j['measured_value']?.toString(),
    measureClass: (j['measure_class'] ?? 'A').toString(),
    overlayPoints: (j['overlay_points'] as List? ?? [])
        .whereType<List>()
        .map<List<int>>(
          (p) => p.whereType<num>().map((n) => n.toInt()).toList(),
        )
        .toList(),
    notScoredReason: j['not_scored_reason']?.toString(),
    ci: j['ci']?.toString(),
    view: j['view']?.toString(),
  );
}

/// The full result of scoring one capture session.
///
/// Everything here is parsed defensively. The server is allowed to add fields
/// and to send null for anything it could not determine - a phone in a village
/// must not hard-fail its decode because one optional key was absent.
class ScoringResult {
  final String animalId;
  final String species;
  final String breedRegistered;

  /// NOT CHECKED, not "wrong". The exact-breed head measured 38.1%
  /// source-held-out and disables itself, so this is false on every record.
  /// Never render it as a mismatch - use [breedVerifyStatus].
  final bool? breedVerified;

  /// `unverified` | `agree` | `disagree`. Only `disagree` is a finding.
  final String breedVerifyStatus;

  /// Which engine answered: `ml-pipeline` (measured) or `baseline`
  /// (demonstration placeholders). This is the single most important field
  /// on the response, and the reason [isDemonstration] exists.
  final String engine;

  // --- breed identity, from the group head (80.2% source-held-out) --------
  final String? predictedSpecies;
  final double? speciesConfidence;
  final bool? speciesConsistent;
  final String? predictedGroup;
  final double? groupConfidence;
  final bool? groupConsistent;

  /// False means that group's own measured recall is poor, or confidence fell
  /// under the measured threshold: show as a hint, never as a finding.
  final bool? groupReliable;

  final bool eligible;
  final String eligibleReason;
  final List<Trait> traits;
  final Map<String, dynamic> weightKg;
  final List<dynamic> symptomVector;
  final List<dynamic> riskReport;
  final List<dynamic> herdAlerts;
  final List<String> healthFlags;

  /// False means no illness screening ran at all. An empty symptom list is
  /// NOT a clean bill of health.
  final bool vetScreened;

  final Map<String, dynamic> reports;
  final String? sessionId;
  final String capturedAt;
  bool synced;

  ScoringResult({
    required this.animalId,
    required this.species,
    required this.breedRegistered,
    required this.breedVerified,
    this.breedVerifyStatus = 'unverified',
    this.engine = 'baseline',
    this.predictedSpecies,
    this.speciesConfidence,
    this.speciesConsistent,
    this.predictedGroup,
    this.groupConfidence,
    this.groupConsistent,
    this.groupReliable,
    required this.eligible,
    required this.eligibleReason,
    required this.traits,
    required this.weightKg,
    required this.symptomVector,
    required this.riskReport,
    this.herdAlerts = const [],
    required this.healthFlags,
    this.vetScreened = false,
    this.reports = const {},
    this.sessionId,
    required this.capturedAt,
    this.synced = false,
  });

  /// True when these numbers were NOT measured from the photographs.
  ///
  /// The baseline engine does not merely fill in blanks - it invents all
  /// twenty scores, a weight, and symptoms. Any screen that shows those
  /// figures has to say so, or a farmer sells an animal on an invention.
  bool get isDemonstration => engine != 'ml-pipeline';

  int get scoredCount => traits.where((t) => t.isScored).length;

  /// Weight rendered as a range, e.g. `173-274 kg`, or null when refused.
  String? get weightRange {
    final lo = (weightKg['low'] as num?)?.toDouble();
    final hi = (weightKg['high'] as num?)?.toDouble();
    if (lo == null || hi == null) return null;
    return '${lo.round()}-${hi.round()} kg';
  }

  String? get weightMethod => weightKg['method']?.toString();

  /// The independent second estimate, which may openly disagree. That
  /// disagreement is information, and is shown rather than hidden.
  String? get weightCrossCheck => weightKg['cross_check']?.toString();

  String? get farmerReport => reports['farmer']?.toString();
  String? get vetReport => reports['vet']?.toString();

  factory ScoringResult.fromJson(Map<String, dynamic> j) => ScoringResult(
    animalId: (j['animal_id'] ?? '').toString(),
    species: (j['species'] ?? '').toString(),
    breedRegistered: (j['breed_registered'] ?? 'Unknown').toString(),
    breedVerified: j['breed_verified'] as bool?,
    breedVerifyStatus: (j['breed_verify_status'] ?? 'unverified').toString(),
    engine: (j['engine'] ?? 'baseline').toString(),
    predictedSpecies: j['predicted_species']?.toString(),
    speciesConfidence: (j['species_confidence'] as num?)?.toDouble(),
    speciesConsistent: j['species_consistent'] as bool?,
    predictedGroup: j['predicted_group']?.toString(),
    groupConfidence: (j['group_confidence'] as num?)?.toDouble(),
    groupConsistent: j['group_consistent'] as bool?,
    groupReliable: j['group_reliable'] as bool?,
    eligible: j['eligible'] as bool? ?? true,
    eligibleReason: (j['eligible_reason'] ?? '').toString(),
    traits: (j['traits'] as List? ?? [])
        .whereType<Map>()
        .map((t) => Trait.fromJson(Map<String, dynamic>.from(t)))
        .toList(),
    weightKg: Map<String, dynamic>.from(j['weight_kg'] as Map? ?? {}),
    symptomVector: j['symptom_vector'] as List? ?? const [],
    riskReport: j['risk_report'] as List? ?? const [],
    herdAlerts: j['herd_alerts'] as List? ?? const [],
    healthFlags: (j['health_flags'] as List? ?? [])
        .map((e) => e.toString())
        .toList(),
    vetScreened: j['vet_screened'] as bool? ?? false,
    reports: Map<String, dynamic>.from(j['reports'] as Map? ?? {}),
    sessionId: j['session_id']?.toString(),
    capturedAt: (j['captured_at'] ?? '').toString(),
    synced: j['synced'] as bool? ?? false,
  );
}
