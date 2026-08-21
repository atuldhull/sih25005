import 'package:flutter/foundation.dart';

import 'api_service.dart';
import 'db_service.dart';

/// The handful of things a user is allowed to change, kept in SQLite.
///
/// There is deliberately no new package here. The app already carries a
/// database; adding shared_preferences for two strings would mean another
/// plugin to build against, and the settings have to survive alongside the
/// queued sessions anyway.
class SettingsService {
  static const String _serverUrlKey = 'server_url';

  /// Emits the current server address so Settings can show it and the rest of
  /// the app can react without reading SQLite on every build.
  static final ValueNotifier<String> serverUrl = ValueNotifier<String>(
    ApiService.defaultBaseUrl,
  );

  /// Loads any saved overrides and applies them.
  ///
  /// Called once before the first request. Never throws: a corrupt or missing
  /// settings row must leave the app on its defaults rather than stop it
  /// starting.
  static Future<void> restore() async {
    try {
      final saved = await DbService.getSetting(_serverUrlKey);
      if (saved != null && saved.trim().isNotEmpty) {
        ApiService.baseUrl = saved.trim();
        serverUrl.value = saved.trim();
      }
    } catch (e) {
      debugPrint('SettingsService.restore failed: $e');
    }
  }

  /// Points the app at a different server and remembers the choice.
  ///
  /// Returns an error message when [raw] is not a usable http(s) address, or
  /// null when it was accepted. Validating here rather than at the first
  /// request means a typo shows up under the text field instead of as a
  /// mysterious upload failure ten minutes later in front of a judge.
  static Future<String?> setServerUrl(String raw) async {
    var url = raw.trim();
    if (url.isEmpty) return 'Enter a server address.';
    if (url.endsWith('/')) url = url.substring(0, url.length - 1);

    final parsed = Uri.tryParse(url);
    if (parsed == null ||
        !parsed.hasScheme ||
        (parsed.scheme != 'http' && parsed.scheme != 'https') ||
        parsed.host.isEmpty) {
      return 'Use a full address, e.g. http://192.168.1.7:8000';
    }

    try {
      await DbService.setSetting(_serverUrlKey, url);
    } catch (e) {
      debugPrint('SettingsService.setServerUrl failed to persist: $e');
      return 'Could not save that address.';
    }

    ApiService.baseUrl = url;
    serverUrl.value = url;
    return null;
  }

  /// Restores the built-in emulator address.
  static Future<void> resetServerUrl() async {
    await setServerUrl(ApiService.defaultBaseUrl);
  }
}
