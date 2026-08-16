class Trait {
  final String name;
  final String category;
  final int? score;
  final double? confidence;
  final String? measuredValue;
  final String measureClass;
  final List<List<int>> overlayPoints;
  final String? notScoredReason;

  Trait({
    required this.name,
    required this.category,
    this.score,
    this.confidence,
    this.measuredValue,
    required this.measureClass,
    this.overlayPoints = const [],
    this.notScoredReason,
  });

  factory Trait.fromJson(Map<String, dynamic> j) => Trait(
    name: j['name'],
    category: j['category'],
    score: j['score'],
    confidence: (j['confidence'] as num?)?.toDouble(),
    measuredValue: j['measured_value'],
    measureClass: j['measure_class'] ?? 'A',
    overlayPoints: (j['overlay_points'] as List? ?? [])
        .map<List<int>>((p) => List<int>.from(p))
        .toList(),
    notScoredReason: j['not_scored_reason'],
  );
}

class ScoringResult {
  final String animalId;
  final String species;
  final String breedRegistered;
  final bool breedVerified;
  final bool eligible;
  final String eligibleReason;
  final List<Trait> traits;
  final Map<String, dynamic> weightKg;
  final List<dynamic> symptomVector;
  final List<dynamic> riskReport;
  final List<String> healthFlags;
  final String capturedAt;
  bool synced;

  ScoringResult({
    required this.animalId,
    required this.species,
    required this.breedRegistered,
    required this.breedVerified,
    required this.eligible,
    required this.eligibleReason,
    required this.traits,
    required this.weightKg,
    required this.symptomVector,
    required this.riskReport,
    required this.healthFlags,
    required this.capturedAt,
    this.synced = false,
  });

  factory ScoringResult.fromJson(Map<String, dynamic> j) => ScoringResult(
    animalId: j['animal_id'],
    species: j['species'],
    breedRegistered: j['breed_registered'],
    breedVerified: j['breed_verified'],
    eligible: j['eligible'],
    eligibleReason: j['eligible_reason'],
    traits: (j['traits'] as List).map((t) => Trait.fromJson(t)).toList(),
    weightKg: j['weight_kg'] ?? {},
    symptomVector: j['symptom_vector'] ?? [],
    riskReport: j['risk_report'] ?? [],
    healthFlags: List<String>.from(j['health_flags'] ?? []),
    capturedAt: j['captured_at'],
    synced: j['synced'] ?? false,
  );
}
