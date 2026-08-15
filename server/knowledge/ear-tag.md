# The ear tag: identity, ruler, and fraud check

## Pashu Aadhaar identity
Every registered animal wears a yellow government ear tag carrying a unique
12-digit Pashu Aadhaar number (also as a barcode). The app reads it straight
from the photo, which links the animal to its official record - breed, birth
date, lactation, calving date - without typing anything.

## The tag as a measuring ruler
The tag's printed parts have government-specified physical sizes: the round
button is 27 mm across, the barcode line is 10 mm tall and the digit line is
18 mm tall. Because those true sizes are known, finding the tag in a photo
tells the app how many millimetres each pixel represents at the animal's
distance - a calibration ruler that is present in every photo for free. The
overall tag panel is NOT used as the ruler because its size varies slightly
between vendors.

## Breed verification
The registered breed comes from the database via the tag - the app does not
need to guess it. Instead, the vision model checks whether the animal in the
photo matches its registered breed. A mismatch flags a possible tag swap or
registration error, which protects the national database's quality.
