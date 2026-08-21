import 'package:flutter/material.dart';

import '../../services/api_service.dart';
import '../../services/capture_source_service.dart';
import '../../services/demo_camera_config.dart';
import '../../services/settings_service.dart';
import '../../services/sync_service.dart';

/// Where the app points, what is still waiting to be sent, and what the app
/// is currently pretending about.
///
/// The last of those is the reason this screen exists rather than being a
/// placeholder: demo camera mode substitutes bundled photographs for the
/// camera, and there was no way to tell from inside the app that it was on.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final ApiService _api = ApiService();

  /// Owned by the screen, not by the dialog.
  ///
  /// It was created inside _editServerUrl and disposed on the line after
  /// `await showDialog(...)`. That await completes when the dialog is POPPED,
  /// while its exit animation is still running and the TextField is still
  /// mounted and still listening - so disposing there tripped
  /// 'package:flutter/src/widgets/framework.dart': Failed assertion:
  /// '_dependents.isEmpty': is not true, and took the whole app to a red
  /// screen. Reproduced on the emulator by saving a server address.
  final TextEditingController _urlController = TextEditingController();

  String? _pingResult;
  bool _pingOk = false;
  bool _pinging = false;
  bool _syncing = false;

  @override
  void initState() {
    super.initState();
    SyncService.instance.refreshPending();
    _testConnection();
  }

  @override
  void dispose() {
    _urlController.dispose();
    _api.close();
    super.dispose();
  }

  Future<void> _testConnection() async {
    setState(() {
      _pinging = true;
      _pingResult = null;
    });

    final reply = await _api.ping();
    if (!mounted) return;

    final error = reply['error'];
    setState(() {
      _pinging = false;
      if (error != null) {
        _pingOk = false;
        _pingResult = error.toString();
        return;
      }
      _pingOk = true;
      final engine = reply['scoring_engine'];
      final real =
          engine is Map && engine['real_pipeline_importable'] == true;
      _pingResult = real
          ? 'Connected. The measurement pipeline is loaded, so sessions are '
                'scored for real.'
          : 'Connected, but the server could not load the measurement '
                'pipeline — it will answer with demonstration placeholders.';
    });
  }

  Future<void> _syncNow() async {
    setState(() => _syncing = true);
    await SyncService.instance.syncPendingSessions();
    if (!mounted) return;
    setState(() => _syncing = false);

    final left = SyncService.instance.pending.value;
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          content: Text(
            left == 0
                ? 'Everything has been sent.'
                : '$left capture${left == 1 ? '' : 's'} still waiting — the '
                      'server could not be reached.',
          ),
        ),
      );
  }

  Future<void> _editServerUrl() async {
    _urlController.text = SettingsService.serverUrl.value;
    final controller = _urlController;
    String? error;

    final saved = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => StatefulBuilder(
        builder: (dialogContext, setDialogState) => AlertDialog(
          title: const Text('Server address'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'On the emulator this is 10.0.2.2. On a real phone, use the '
                'laptop’s address on the same Wi-Fi.',
                style: TextStyle(fontSize: 13, height: 1.35),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: controller,
                autofocus: true,
                keyboardType: TextInputType.url,
                decoration: InputDecoration(
                  border: const OutlineInputBorder(),
                  hintText: 'http://192.168.1.7:8000',
                  errorText: error,
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () async {
                await SettingsService.resetServerUrl();
                if (dialogContext.mounted) {
                  Navigator.of(dialogContext).pop(true);
                }
              },
              child: const Text('Reset'),
            ),
            TextButton(
              onPressed: () => Navigator.of(dialogContext).pop(false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () async {
                final problem = await SettingsService.setServerUrl(
                  controller.text,
                );
                if (problem != null) {
                  setDialogState(() => error = problem);
                  return;
                }
                if (dialogContext.mounted) {
                  Navigator.of(dialogContext).pop(true);
                }
              },
              child: const Text('Save'),
            ),
          ],
        ),
      ),
    );

    if (saved == true && mounted) {
      setState(() {});
      await _testConnection();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        children: [
          _SectionLabel('Server'),

          ValueListenableBuilder<String>(
            valueListenable: SettingsService.serverUrl,
            builder: (context, url, _) => ListTile(
              leading: const Icon(Icons.dns_outlined),
              title: const Text('Server address'),
              subtitle: Text(url),
              trailing: const Icon(Icons.edit_outlined, size: 20),
              onTap: _editServerUrl,
            ),
          ),

          ListTile(
            leading: _pinging
                ? const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : Icon(
                    _pingOk ? Icons.check_circle_outline : Icons.error_outline,
                    color: _pingOk
                        ? const Color(0xFF2E7D32)
                        : const Color(0xFFD32F2F),
                  ),
            title: const Text('Connection'),
            subtitle: Text(
              _pinging ? 'Checking…' : (_pingResult ?? 'Not checked yet.'),
            ),
            trailing: TextButton(
              onPressed: _pinging ? null : _testConnection,
              child: const Text('Test'),
            ),
          ),

          const Divider(height: 24),
          _SectionLabel('Captures waiting to be sent'),

          ValueListenableBuilder<int>(
            valueListenable: SyncService.instance.pending,
            builder: (context, count, _) => ListTile(
              leading: Icon(
                count == 0 ? Icons.cloud_done_outlined : Icons.cloud_upload_outlined,
                color: count == 0 ? const Color(0xFF2E7D32) : null,
              ),
              title: Text(
                count == 0
                    ? 'Nothing waiting'
                    : '$count capture${count == 1 ? '' : 's'} waiting',
              ),
              subtitle: const Text(
                'A capture is never deleted. It is kept until the server '
                'confirms it.',
              ),
              trailing: _syncing
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : TextButton(
                      onPressed: count == 0 ? null : _syncNow,
                      child: const Text('Send now'),
                    ),
            ),
          ),

          const Divider(height: 24),
          _SectionLabel('Capture'),

          ValueListenableBuilder<CaptureSource>(
            valueListenable: CaptureSourceService.source,
            builder: (context, source, _) {
              final demo = source == CaptureSource.demo;
              return Column(
                children: [
                  SwitchListTile(
                    value: !demo,
                    onChanged: (useCamera) => CaptureSourceService.set(
                      useCamera ? CaptureSource.camera : CaptureSource.demo,
                    ),
                    secondary: Icon(
                      demo
                          ? Icons.photo_library_outlined
                          : Icons.camera_alt_outlined,
                      color: demo ? const Color(0xFFEF6C00) : null,
                    ),
                    title: Text(
                      demo ? 'Demo camera mode is ON' : 'Using the real camera',
                    ),
                    subtitle: Text(
                      demo
                          ? 'Capture screens show bundled photographs instead '
                                'of a camera, so this runs on an emulator. The '
                                'scorecard is produced from those photographs, '
                                'not from the animal in front of you.'
                          : 'Every capture screen opens the live camera. If '
                                'the preview stays black, the camera '
                                'permission was declined - switch back and '
                                'grant it in Android settings.',
                      style: const TextStyle(height: 1.35),
                    ),
                    isThreeLine: true,
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(72, 0, 16, 12),
                    child: Row(
                      children: [
                        Icon(
                          Icons.photo_outlined,
                          size: 16,
                          color: Colors.grey.shade600,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Choosing an existing photo works in both modes — '
                            'every capture screen has a Gallery button.',
                            style: TextStyle(
                              fontSize: 12.5,
                              height: 1.3,
                              color: Colors.grey.shade600,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              );
            },
          ),

          ListTile(
            leading: const Icon(Icons.timer_outlined),
            title: const Text('Walking video length'),
            subtitle: Text('${DemoCameraConfig.videoDurationSeconds} seconds'),
          ),

          const Divider(height: 24),
          _SectionLabel('About'),

          const ListTile(
            leading: Icon(Icons.info_outline),
            title: Text('Pashu Scorer'),
            subtitle: Text(
              'Image-based animal type classification for cattle and '
              'buffaloes. Scores are biological measurements, not quality '
              'judgements.',
            ),
            isThreeLine: true,
          ),

          const SizedBox(height: 24),
        ],
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final String text;

  const _SectionLabel(this.text);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
      child: Text(
        text.toUpperCase(),
        style: TextStyle(
          fontSize: 11.5,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.8,
          color: Theme.of(context).colorScheme.primary,
        ),
      ),
    );
  }
}
