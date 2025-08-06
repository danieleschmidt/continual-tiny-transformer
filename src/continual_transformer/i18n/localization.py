"""Internationalization and localization support for continual transformers."""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import locale
import gettext
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class LocaleInfo:
    """Locale information with regional settings."""
    code: str  # Language code (e.g., 'en', 'es', 'zh')
    name: str  # Display name
    country: Optional[str] = None  # Country code (e.g., 'US', 'MX', 'CN')
    direction: str = "ltr"  # Text direction: 'ltr' or 'rtl'
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    decimal_separator: str = "."
    thousands_separator: str = ","
    currency_symbol: str = "$"
    currency_position: str = "before"  # 'before' or 'after'


@dataclass
class TranslationEntry:
    """Translation entry with metadata."""
    key: str
    text: str
    context: Optional[str] = None
    plural_forms: Dict[str, str] = field(default_factory=dict)
    last_modified: Optional[datetime] = None
    translator: Optional[str] = None
    reviewed: bool = False


class I18nManager:
    """Comprehensive internationalization manager."""
    
    SUPPORTED_LOCALES = {
        'en': LocaleInfo('en', 'English', 'US', 'ltr', '%m/%d/%Y', '%I:%M:%S %p', '.', ',', '$', 'before'),
        'es': LocaleInfo('es', 'Español', 'ES', 'ltr', '%d/%m/%Y', '%H:%M:%S', ',', '.', '€', 'after'),
        'fr': LocaleInfo('fr', 'Français', 'FR', 'ltr', '%d/%m/%Y', '%H:%M:%S', ',', ' ', '€', 'after'),
        'de': LocaleInfo('de', 'Deutsch', 'DE', 'ltr', '%d.%m.%Y', '%H:%M:%S', ',', '.', '€', 'after'),
        'ja': LocaleInfo('ja', '日本語', 'JP', 'ltr', '%Y/%m/%d', '%H:%M:%S', '.', ',', '¥', 'before'),
        'zh': LocaleInfo('zh', '中文', 'CN', 'ltr', '%Y年%m月%d日', '%H:%M:%S', '.', ',', '¥', 'before'),
        'ar': LocaleInfo('ar', 'العربية', 'SA', 'rtl', '%d/%m/%Y', '%H:%M:%S', '.', ',', 'ر.س', 'after'),
        'pt': LocaleInfo('pt', 'Português', 'BR', 'ltr', '%d/%m/%Y', '%H:%M:%S', ',', '.', 'R$', 'before'),
        'it': LocaleInfo('it', 'Italiano', 'IT', 'ltr', '%d/%m/%Y', '%H:%M:%S', ',', '.', '€', 'after'),
        'ru': LocaleInfo('ru', 'Русский', 'RU', 'ltr', '%d.%m.%Y', '%H:%M:%S', ',', ' ', '₽', 'after'),
        'ko': LocaleInfo('ko', '한국어', 'KR', 'ltr', '%Y.%m.%d', '%H:%M:%S', '.', ',', '₩', 'before'),
        'hi': LocaleInfo('hi', 'हिन्दी', 'IN', 'ltr', '%d/%m/%Y', '%H:%M:%S', '.', ',', '₹', 'before')
    }
    
    def __init__(self, config=None, default_locale: str = 'en'):
        self.config = config
        self.default_locale = default_locale
        self.current_locale = default_locale
        
        # Translation storage
        self.translations: Dict[str, Dict[str, TranslationEntry]] = {}
        self.translation_cache: Dict[str, str] = {}
        
        # Translation directory
        self.translations_dir = Path(__file__).parent / "translations"
        self.translations_dir.mkdir(exist_ok=True)
        
        # Load default translations
        self._load_base_translations()
        
        # Auto-detect locale if available
        self._auto_detect_locale()
        
        logger.info(f"I18n manager initialized with locale: {self.current_locale}")
    
    def _auto_detect_locale(self):
        """Auto-detect system locale."""
        try:
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                lang_code = system_locale.split('_')[0]
                if lang_code in self.SUPPORTED_LOCALES:
                    self.current_locale = lang_code
                    logger.info(f"Auto-detected locale: {lang_code}")
        except Exception as e:
            logger.warning(f"Could not auto-detect locale: {e}")
    
    def set_locale(self, locale_code: str) -> bool:
        """Set current locale."""
        if locale_code not in self.SUPPORTED_LOCALES:
            logger.error(f"Unsupported locale: {locale_code}")
            return False
        
        self.current_locale = locale_code
        self.translation_cache.clear()  # Clear cache on locale change
        
        # Try to set system locale
        try:
            locale_info = self.SUPPORTED_LOCALES[locale_code]
            if locale_info.country:
                locale_name = f"{locale_code}_{locale_info.country}"
            else:
                locale_name = locale_code
                
            locale.setlocale(locale.LC_ALL, f"{locale_name}.UTF-8")
        except locale.Error:
            # Fallback to C locale
            logger.warning(f"Could not set system locale to {locale_code}, using default")
        
        logger.info(f"Locale changed to: {locale_code}")
        return True
    
    def get_current_locale(self) -> LocaleInfo:
        """Get current locale information."""
        return self.SUPPORTED_LOCALES[self.current_locale]
    
    def get_supported_locales(self) -> Dict[str, LocaleInfo]:
        """Get all supported locales."""
        return self.SUPPORTED_LOCALES.copy()
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a key with optional formatting parameters."""
        cache_key = f"{self.current_locale}:{key}:{str(kwargs)}"
        
        # Check cache first
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Get translation
        translation = self._get_translation(key)
        
        # Apply formatting if parameters provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation formatting failed for key '{key}': {e}")
        
        # Cache the result
        self.translation_cache[cache_key] = translation
        return translation
    
    def translate_plural(self, key: str, count: int, **kwargs) -> str:
        """Translate with plural forms support."""
        # Get the plural form based on count and locale rules
        plural_form = self._get_plural_form(count)
        plural_key = f"{key}#{plural_form}"
        
        # Try plural form first, fallback to singular
        translation = self._get_translation(plural_key)
        if translation == plural_key:  # Not found, try singular
            translation = self._get_translation(key)
        
        # Add count to formatting parameters
        kwargs['count'] = count
        
        # Apply formatting
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Plural translation formatting failed for key '{key}': {e}")
        
        return translation
    
    def _get_translation(self, key: str) -> str:
        """Get translation for a key in current locale."""
        # Try current locale
        if self.current_locale in self.translations:
            if key in self.translations[self.current_locale]:
                return self.translations[self.current_locale][key].text
        
        # Fallback to default locale
        if (self.default_locale != self.current_locale and 
            self.default_locale in self.translations):
            if key in self.translations[self.default_locale]:
                return self.translations[self.default_locale][key].text
        
        # Return key if no translation found
        logger.warning(f"Translation not found for key: {key}")
        return key
    
    def _get_plural_form(self, count: int) -> str:
        """Get plural form based on count and locale rules."""
        # Simplified plural rules - in reality, would use more sophisticated logic
        locale_info = self.get_current_locale()
        
        # Basic rules for common locales
        if locale_info.code in ['en']:
            return 'one' if count == 1 else 'other'
        elif locale_info.code in ['es', 'fr', 'de', 'it', 'pt']:
            return 'one' if count <= 1 else 'other'
        elif locale_info.code in ['ru']:
            if count % 10 == 1 and count % 100 != 11:
                return 'one'
            elif count % 10 in [2, 3, 4] and count % 100 not in [12, 13, 14]:
                return 'few'
            else:
                return 'many'
        elif locale_info.code in ['zh', 'ja', 'ko']:
            return 'other'  # No plural distinction
        else:
            return 'one' if count == 1 else 'other'
    
    def format_date(self, date: datetime, format_type: str = 'short') -> str:
        """Format date according to locale settings."""
        locale_info = self.get_current_locale()
        
        if format_type == 'short':
            return date.strftime(locale_info.date_format)
        elif format_type == 'long':
            # Long format with month names (would need locale-specific month names)
            return date.strftime('%B %d, %Y')
        elif format_type == 'time':
            return date.strftime(locale_info.time_format)
        elif format_type == 'datetime':
            return f"{date.strftime(locale_info.date_format)} {date.strftime(locale_info.time_format)}"
        else:
            return date.strftime(locale_info.date_format)
    
    def format_number(self, number: Union[int, float], decimal_places: Optional[int] = None) -> str:
        """Format number according to locale settings."""
        locale_info = self.get_current_locale()
        
        if decimal_places is not None:
            number_str = f"{number:.{decimal_places}f}"
        else:
            number_str = str(number)
        
        # Split into integer and decimal parts
        if '.' in number_str:
            integer_part, decimal_part = number_str.split('.')
        else:
            integer_part, decimal_part = number_str, None
        
        # Add thousands separators
        if len(integer_part) > 3:
            # Add separators every 3 digits from right
            formatted_integer = ""
            for i, digit in enumerate(reversed(integer_part)):
                if i > 0 and i % 3 == 0:
                    formatted_integer = locale_info.thousands_separator + formatted_integer
                formatted_integer = digit + formatted_integer
        else:
            formatted_integer = integer_part
        
        # Combine with decimal part
        if decimal_part:
            return formatted_integer + locale_info.decimal_separator + decimal_part
        else:
            return formatted_integer
    
    def format_currency(self, amount: Union[int, float], decimal_places: int = 2) -> str:
        """Format currency according to locale settings."""
        locale_info = self.get_current_locale()
        formatted_amount = self.format_number(amount, decimal_places)
        
        if locale_info.currency_position == 'before':
            return f"{locale_info.currency_symbol}{formatted_amount}"
        else:
            return f"{formatted_amount} {locale_info.currency_symbol}"
    
    def load_translations_from_file(self, file_path: str, locale_code: str):
        """Load translations from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations_data = json.load(f)
            
            if locale_code not in self.translations:
                self.translations[locale_code] = {}
            
            # Convert to TranslationEntry objects
            for key, value in translations_data.items():
                if isinstance(value, str):
                    entry = TranslationEntry(key=key, text=value)
                elif isinstance(value, dict):
                    entry = TranslationEntry(
                        key=key,
                        text=value.get('text', key),
                        context=value.get('context'),
                        plural_forms=value.get('plural_forms', {}),
                        translator=value.get('translator'),
                        reviewed=value.get('reviewed', False)
                    )
                else:
                    continue
                
                self.translations[locale_code][key] = entry
            
            logger.info(f"Loaded {len(translations_data)} translations for {locale_code}")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load translations from {file_path}: {e}")
    
    def save_translations_to_file(self, file_path: str, locale_code: str):
        """Save translations to JSON file."""
        if locale_code not in self.translations:
            logger.warning(f"No translations found for locale: {locale_code}")
            return
        
        try:
            translations_data = {}
            for key, entry in self.translations[locale_code].items():
                translations_data[key] = {
                    'text': entry.text,
                    'context': entry.context,
                    'plural_forms': entry.plural_forms,
                    'translator': entry.translator,
                    'reviewed': entry.reviewed,
                    'last_modified': entry.last_modified.isoformat() if entry.last_modified else None
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(translations_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(translations_data)} translations to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save translations to {file_path}: {e}")
    
    def add_translation(self, key: str, text: str, locale_code: Optional[str] = None, 
                       context: Optional[str] = None, plural_forms: Optional[Dict[str, str]] = None):
        """Add or update a translation."""
        if locale_code is None:
            locale_code = self.current_locale
        
        if locale_code not in self.translations:
            self.translations[locale_code] = {}
        
        entry = TranslationEntry(
            key=key,
            text=text,
            context=context,
            plural_forms=plural_forms or {},
            last_modified=datetime.now()
        )
        
        self.translations[locale_code][key] = entry
        
        # Clear cache for this key
        cache_keys_to_remove = [k for k in self.translation_cache.keys() if k.startswith(f"{locale_code}:{key}")]
        for cache_key in cache_keys_to_remove:
            del self.translation_cache[cache_key]
    
    def get_translation_status(self) -> Dict[str, Dict[str, Any]]:
        """Get translation status for all locales."""
        status = {}
        
        # Get all keys from default locale
        default_keys = set()
        if self.default_locale in self.translations:
            default_keys = set(self.translations[self.default_locale].keys())
        
        for locale_code in self.SUPPORTED_LOCALES:
            locale_keys = set()
            if locale_code in self.translations:
                locale_keys = set(self.translations[locale_code].keys())
            
            missing_keys = default_keys - locale_keys
            extra_keys = locale_keys - default_keys
            
            status[locale_code] = {
                'total_keys': len(locale_keys),
                'missing_keys': len(missing_keys),
                'extra_keys': len(extra_keys),
                'completion_rate': len(locale_keys) / len(default_keys) if default_keys else 1.0,
                'missing_key_list': list(missing_keys),
                'extra_key_list': list(extra_keys)
            }
        
        return status
    
    def _load_base_translations(self):
        """Load base translations for essential messages."""
        base_translations = {
            'en': {
                'error.general': 'An error occurred',
                'error.not_found': 'Item not found',
                'error.invalid_input': 'Invalid input provided',
                'error.permission_denied': 'Permission denied',
                'success.saved': 'Successfully saved',
                'success.deleted': 'Successfully deleted',
                'success.updated': 'Successfully updated',
                'model.training': 'Training model...',
                'model.inference': 'Running inference...',
                'task.completed': 'Task completed successfully',
                'task.failed': 'Task failed',
                'validation.required': 'This field is required',
                'validation.invalid_format': 'Invalid format',
                'settings.language': 'Language',
                'settings.region': 'Region',
                'common.yes': 'Yes',
                'common.no': 'No',
                'common.ok': 'OK',
                'common.cancel': 'Cancel',
                'common.save': 'Save',
                'common.delete': 'Delete',
                'common.edit': 'Edit',
                'common.view': 'View',
                'common.loading': 'Loading...',
                'date.today': 'Today',
                'date.yesterday': 'Yesterday',
                'date.tomorrow': 'Tomorrow',
                'time.seconds_ago': '{count} seconds ago',
                'time.minutes_ago#one': '1 minute ago',
                'time.minutes_ago#other': '{count} minutes ago',
                'time.hours_ago#one': '1 hour ago',
                'time.hours_ago#other': '{count} hours ago'
            }
        }
        
        # Load English translations
        if 'en' not in self.translations:
            self.translations['en'] = {}
        
        for key, text in base_translations['en'].items():
            self.translations['en'][key] = TranslationEntry(key=key, text=text)
        
        # Load locale-specific translations from files if they exist
        for locale_code in self.SUPPORTED_LOCALES:
            translation_file = self.translations_dir / f"{locale_code}.json"
            if translation_file.exists():
                self.load_translations_from_file(str(translation_file), locale_code)
    
    def extract_translatable_strings(self, source_dir: str) -> Set[str]:
        """Extract translatable strings from source code."""
        translatable_strings = set()
        
        # Pattern to match translation function calls
        import re
        translation_pattern = re.compile(r'(?:t|translate|_)\s*\(\s*["\']([^"\']+)["\']')
        
        for py_file in Path(source_dir).rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                matches = translation_pattern.findall(content)
                translatable_strings.update(matches)
                
            except (OSError, UnicodeDecodeError):
                continue
        
        return translatable_strings
    
    def generate_translation_template(self, output_file: str, source_dir: Optional[str] = None):
        """Generate translation template file with all translatable strings."""
        if source_dir:
            keys = self.extract_translatable_strings(source_dir)
        else:
            # Use existing keys from default locale
            keys = set(self.translations.get(self.default_locale, {}).keys())
        
        template = {}
        for key in sorted(keys):
            template[key] = {
                'text': '',  # Empty text to be filled by translator
                'context': '',
                'plural_forms': {},
                'translator': '',
                'reviewed': False
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated translation template with {len(template)} keys: {output_file}")


# Global i18n instance
_i18n_instance = None

def get_i18n() -> I18nManager:
    """Get global i18n instance."""
    global _i18n_instance
    if _i18n_instance is None:
        _i18n_instance = I18nManager()
    return _i18n_instance

def init_i18n(config=None, default_locale: str = 'en') -> I18nManager:
    """Initialize global i18n instance."""
    global _i18n_instance
    _i18n_instance = I18nManager(config, default_locale)
    return _i18n_instance

# Convenience functions
def t(key: str, **kwargs) -> str:
    """Translate a key (convenience function)."""
    return get_i18n().translate(key, **kwargs)

def tn(key: str, count: int, **kwargs) -> str:
    """Translate with plural forms (convenience function)."""
    return get_i18n().translate_plural(key, count, **kwargs)

def set_locale(locale_code: str) -> bool:
    """Set current locale (convenience function)."""
    return get_i18n().set_locale(locale_code)

def get_locale() -> LocaleInfo:
    """Get current locale info (convenience function)."""
    return get_i18n().get_current_locale()

# Decorator for translatable functions
def translatable(func):
    """Decorator to mark functions as having translatable strings."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper._translatable = True
    return wrapper

# Context manager for temporary locale changes
class temp_locale:
    """Context manager for temporary locale changes."""
    
    def __init__(self, locale_code: str):
        self.locale_code = locale_code
        self.original_locale = None
    
    def __enter__(self):
        i18n = get_i18n()
        self.original_locale = i18n.current_locale
        i18n.set_locale(self.locale_code)
        return i18n
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_locale:
            get_i18n().set_locale(self.original_locale)