\version "2.24.0"
% MIDI-only score of Vivaldi "Spring" mvt. I (Mutopia parts, unchanged notes), quarter = 60 so 1 MIDI second = 1 beat
\include "spring1.ly"
\include "spring1a.ly"
\include "spring1b.ly"
\include "spring1c.ly"
\include "spring1d.ly"
\score {
  <<
    \new Staff \ViolinoPrincipaleSpringOne
    \new Staff \ViolinoPrimoSpringOne
    \new Staff \ViolinoSecondoSpringOne
    \new Staff \AltoViolaSpringOne
    \new Staff \VioloncelloSpringOne
  >>
  \midi { \tempo 4 = 60 }
}
