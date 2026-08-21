import 'package:flutter/material.dart';

import '../../services/animal_cache_service.dart';
import '../../services/api_service.dart';
import '../assistant/chat_screen.dart';
import 'history_screen.dart';

/// Displays an animal's profile with live API data and offline cache fallback.
///
/// Data loading order:
/// 1. Try [ApiService.getAnimal] — if successful, display live data and cache it.
/// 2. If the API fails, read from [AnimalCacheService] — display cached data
///    with an offline indicator.
/// 3. If neither is available, show "No animal data available offline."
class AnimalProfileScreen extends StatefulWidget {
  final String animalId;
  final VoidCallback? onStartScoring;

  const AnimalProfileScreen({
    super.key,
    required this.animalId,
    this.onStartScoring,
  });

  @override
  State<AnimalProfileScreen> createState() => _AnimalProfileScreenState();
}

enum _LoadState { loading, live, cached, noData, error }

class _AnimalProfileScreenState extends State<AnimalProfileScreen> {
  _LoadState _state = _LoadState.loading;
  Map<String, dynamic>? _profile;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _state = _LoadState.loading;
    });

    try {
      final result = await ApiService().getAnimal(widget.animalId);

      if (_isErrorMap(result)) {
        // API failed — try the offline cache.
        final cached = await AnimalCacheService.getAnimal(widget.animalId);
        if (!mounted) return;

        if (cached != null) {
          setState(() {
            _profile = cached;
            _state = _LoadState.cached;
          });
        } else {
          setState(() {
            _profile = null;
            _state = _LoadState.noData;
          });
        }
        return;
      }

      // API succeeded — display live data and cache it for offline use.
      await AnimalCacheService.saveAnimal(widget.animalId, result);
      if (!mounted) return;

      setState(() {
        _profile = result;
        _state = _LoadState.live;
      });
    } catch (_) {
      // Unexpected error — try the offline cache as a last resort.
      try {
        final cached = await AnimalCacheService.getAnimal(widget.animalId);
        if (!mounted) return;

        if (cached != null) {
          setState(() {
            _profile = cached;
            _state = _LoadState.cached;
          });
        } else {
          setState(() {
            _profile = null;
            _state = _LoadState.error;
          });
        }
      } catch (_) {
        if (!mounted) return;
        setState(() {
          _profile = null;
          _state = _LoadState.error;
        });
      }
    }
  }

  /// Detects whether an ApiService result map represents a failure.
  ///
  /// ApiService returns `{'error': ...}` on any failure (network, timeout,
  /// non-2xx, malformed JSON). A real animal profile must NOT contain an
  /// `error` key.
  bool _isErrorMap(Map<String, dynamic> map) {
    final error = map['error'];
    return error is String && error.isNotEmpty;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Animal Profile')),
      body: switch (_state) {
        _LoadState.loading => const Center(child: CircularProgressIndicator()),
        _LoadState.live => _buildProfile(isLive: true),
        _LoadState.cached => _buildProfile(isLive: false),
        _LoadState.noData => const _NoDataView(),
        _LoadState.error => _ErrorView(onRetry: _load),
      },
    );
  }

  Widget _buildProfile({required bool isLive}) {
    final profile = _profile;
    if (profile == null) {
      return _ErrorView(onRetry: _load);
    }

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _CacheStatusBanner(isLive: isLive),
        const SizedBox(height: 16),
        _ProfileCard(animalId: widget.animalId, profile: profile),
        const SizedBox(height: 16),
        _EligibilityCard(profile: profile),
        const SizedBox(height: 24),
        _StartScoringButton(onStartScoring: widget.onStartScoring),
        const SizedBox(height: 12),

        // Everything about this animal, reachable from her own record: her
        // past sessions, and the assistant that answers from them.
        SizedBox(
          width: double.infinity,
          height: 50,
          child: OutlinedButton.icon(
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (_) => HistoryScreen(animalId: widget.animalId),
              ),
            ),
            icon: const Icon(Icons.timeline),
            label: const Text('Scoring History'),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          height: 50,
          child: OutlinedButton.icon(
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (_) => ChatScreen(
                  animalId: widget.animalId,
                  animalLabel: profile['breed']?.toString(),
                ),
              ),
            ),
            icon: const Icon(Icons.chat_bubble_outline),
            label: const Text('Ask about this animal'),
          ),
        ),
        const SizedBox(height: 24),
      ],
    );
  }
}

/// Banner indicating whether the displayed data is live or cached.
class _CacheStatusBanner extends StatelessWidget {
  final bool isLive;

  const _CacheStatusBanner({required this.isLive});

  @override
  Widget build(BuildContext context) {
    final color = isLive ? Colors.green.shade700 : Colors.orange.shade800;
    final icon = isLive ? Icons.cloud_done_outlined : Icons.cloud_off_outlined;
    final label = isLive ? 'LIVE DATA' : 'OFFLINE — LAST CACHED DATA';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 8),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: color,
              letterSpacing: 0.5,
            ),
          ),
        ],
      ),
    );
  }
}

/// Displays the animal's profile fields in a clean field/value layout.
class _ProfileCard extends StatelessWidget {
  final String animalId;
  final Map<String, dynamic> profile;

  const _ProfileCard({required this.animalId, required this.profile});

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
              'Profile Information',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 12),
            _FieldRow(label: 'Animal ID', value: animalId),
            _FieldRow(label: 'Breed', value: _stringValue(profile['breed'])),
            _FieldRow(
              label: 'Age / DOB',
              value:
                  _stringValue(profile['age']) ??
                  _stringValue(profile['dob']) ??
                  _stringValue(profile['date_of_birth']),
            ),
            _FieldRow(
              label: 'Lactation number',
              value:
                  // lactation_no is what GET /animal/{id} actually sends;
                  // without it this row read "-" on every animal, which is
                  // the number the eligibility rule is built on.
                  _stringValue(profile['lactation_no']) ??
                  _stringValue(profile['lactation_number']) ??
                  _stringValue(profile['lactation']),
            ),
            _FieldRow(
              label: 'Last calving date',
              value:
                  _stringValue(profile['last_calving_date']) ??
                  _stringValue(profile['last_calving']),
            ),
          ],
        ),
      ),
    );
  }

  String? _stringValue(dynamic value) {
    if (value == null) return null;
    final text = value.toString().trim();
    return text.isEmpty ? null : text;
  }
}

/// Displays the server-provided eligibility status.
class _EligibilityCard extends StatelessWidget {
  final Map<String, dynamic> profile;

  const _EligibilityCard({required this.profile});

  @override
  Widget build(BuildContext context) {
    final eligible = profile['eligible'];
    final reason = profile['eligible_reason'];

    final bool isEligible = eligible == true;

    final Color color;
    final IconData icon;
    final String title;
    final String? subtitle;

    if (isEligible) {
      color = Colors.green.shade700;
      icon = Icons.check_circle_outline;
      title = 'Eligible for scoring';
      subtitle = null;
    } else {
      color = Colors.orange.shade800;
      icon = Icons.info_outline;
      title = 'Not eligible';
      subtitle = reason is String && reason.isNotEmpty ? reason : null;
    }

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: color.withValues(alpha: 0.4)),
      ),
      color: color.withValues(alpha: 0.06),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: color, size: 24),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      color: color,
                    ),
                  ),
                  if (subtitle != null) ...[
                    const SizedBox(height: 4),
                    Text(
                      subtitle,
                      style: TextStyle(
                        fontSize: 13,
                        color: Colors.grey.shade800,
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Button that triggers the optional scoring-session callback.
class _StartScoringButton extends StatelessWidget {
  final VoidCallback? onStartScoring;

  const _StartScoringButton({this.onStartScoring});

  @override
  Widget build(BuildContext context) {
    final callback = onStartScoring;

    if (callback == null) {
      return SizedBox(
        width: double.infinity,
        child: OutlinedButton.icon(
          onPressed: null,
          icon: const Icon(Icons.qr_code_scanner),
          label: const Text('Start New Scoring Session'),
        ),
      );
    }

    return SizedBox(
      width: double.infinity,
      child: FilledButton.icon(
        onPressed: callback,
        icon: const Icon(Icons.qr_code_scanner),
        label: const Text('Start New Scoring Session'),
      ),
    );
  }
}

/// Field/value row used in the profile card.
class _FieldRow extends StatelessWidget {
  final String label;
  final String? value;

  const _FieldRow({required this.label, this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 140,
            child: Text(
              label,
              style: TextStyle(fontSize: 13, color: Colors.grey.shade600),
            ),
          ),
          Expanded(
            child: Text(
              value ?? '—',
              style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
            ),
          ),
        ],
      ),
    );
  }
}

/// Shown when neither live nor cached data is available.
class _NoDataView extends StatelessWidget {
  const _NoDataView();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.search_off, size: 48, color: Colors.grey.shade400),
            const SizedBox(height: 12),
            const Text(
              'No animal data available offline.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}

/// Shown when an unexpected error occurs and no cache fallback exists.
class _ErrorView extends StatelessWidget {
  final VoidCallback onRetry;

  const _ErrorView({required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.error_outline, size: 48, color: Colors.grey.shade400),
            const SizedBox(height: 12),
            const Text(
              'Unable to load animal data.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 16),
            OutlinedButton(onPressed: onRetry, child: const Text('Retry')),
          ],
        ),
      ),
    );
  }
}
