"""Domain rules shared by the API endpoints and the chatbot."""
from datetime import date, datetime


def check_eligibility(animal: dict, lang: str = "en") -> tuple[bool, str]:
    """NDDB rule: type-classify only in FIRST lactation, day 30-90
    after calving. Refusing to score outside this window is a
    feature, not a limitation - the reason string is shown in-app.
    lang='hi' returns the reason in Hindi for the chatbot."""
    lact = animal["lactation_no"]
    calving = datetime.strptime(animal["last_calving_date"], "%Y-%m-%d").date()
    days = (date.today() - calving).days
    hi = lang == "hi"

    if lact != 1:
        return False, (f"पहली ब्यांत में नहीं है (अभी ब्यांत {lact})" if hi else
                       f"not in first lactation (currently lactation {lact})")
    if days < 30:
        return False, (f"ब्याने के केवल {days} दिन हुए हैं - स्कोरिंग 30वें दिन से हो सकती है" if hi else
                       f"only {days} days since calving - scoring allowed from day 30")
    if days > 90:
        return False, (f"ब्याने के {days} दिन हो चुके हैं - 90 दिन की सीमा निकल गई" if hi else
                       f"{days} days since calving - past the day-90 window")
    return True, (f"पहली ब्यांत, ब्याने के {days}वें दिन" if hi else
                  f"first lactation, day {days} after calving")
