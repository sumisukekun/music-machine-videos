\version "2.24.0"
% MIDI-only score of Vivaldi "Winter" mvt. II Largo (Mutopia parts, unchanged notes), quarter = 60 so 1 MIDI second = 1 beat
\include "winter2.ly"
\include "winter2a.ly"
\include "winter2b.ly"
\include "winter2c.ly"
\include "winter2d.ly"
\score {
  <<
    \new Staff \ViolinoPrincipaleWinterTwo
    \new Staff \ViolinoPrimoWinterTwo
    \new Staff \ViolinoSecondoWinterTwo
    \new Staff \AltoViolaWinterTwo
    \new Staff \VioloncelloWinterTwo
  >>
  \midi { \tempo 4 = 60 }
}
