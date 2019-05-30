generateEnvironment(EnvWidth, EnvHeight, R0X, R0Y, R0W, R0H, R1X, R1Y, R1W, R1H, R2X, R2Y, R2W, R2H, R3X, R3Y, R3W, R3H) :-
repeat, 
random(102.0, 144.5, R0W), 
random(102.0, 144.5, R0H), WSUB0 is EnvWidth - R0W, 
random(0.0, WSUB0, R0X), HSUB0 is EnvHeight - R0H, 
random(0.0, HSUB0, R0Y), random(68.0, 102.0, R1W), 
random(68.0, 102.0, R1H), WSUB1 is EnvWidth - R1W, 
random(0.0, WSUB1, R1X), HSUB1 is EnvHeight - R1H, 
random(0.0, HSUB1, R1Y), random(85.0, 127.5, R2W), 
random(85.0, 127.5, R2H), WSUB2 is EnvWidth - R2W, 
random(0.0, WSUB2, R2X), HSUB2 is EnvHeight - R2H, 
random(0.0, HSUB2, R2Y), random(127.5, 170.0, R3W), 
random(127.5, 170.0, R3H), WSUB3 is EnvWidth - R3W, 
random(0.0, WSUB3, R3X), HSUB3 is EnvHeight - R3H, 
random(0.0, HSUB3, R3Y), 
{(R0X + R0W + 21.25 =< R1X ; R1X + R1W + 21.25 =< R0X) ; 
(R0Y + R0H + 21.25 =< R1Y ; R1Y + R1H + 21.25 =< R0Y)}, 
{(R0X + R0W + 21.25 =< R2X ; R2X + R2W + 21.25 =< R0X) ; 
(R0Y + R0H + 21.25 =< R2Y ; R2Y + R2H + 21.25 =< R0Y)}, 
{(R0X + R0W + 21.25 =< R3X ; R3X + R3W + 21.25 =< R0X) ; 
(R0Y + R0H + 21.25 =< R3Y ; R3Y + R3H + 21.25 =< R0Y)}, 
{(R1X + R1W + 21.25 =< R2X ; R2X + R2W + 21.25 =< R1X) ; 
(R1Y + R1H + 21.25 =< R2Y ; R2Y + R2H + 21.25 =< R1Y)}, 
{(R1X + R1W + 21.25 =< R3X ; R3X + R3W + 21.25 =< R1X) ; 
(R1Y + R1H + 21.25 =< R3Y ; R3Y + R3H + 21.25 =< R1Y)}, 
{(R2X + R2W + 21.25 =< R3X ; R3X + R3W + 21.25 =< R2X) ; 
(R2Y + R2H + 21.25 =< R3Y ; R3Y + R3H + 21.25 =< R2Y)}, 
CentroX is (R0X + R1X + R2X + R3X) / 4, CentroY is (R0Y + R1Y + R2Y + R3Y) / 4, 
DistanzaRoom0 is sqrt(((R0X + R0W/2) - (CentroX))^2 + ((R0Y + R0H/2) - (CentroY))^2), {DistanzaRoom0 =< 187.0}, 
DistanzaRoom1 is sqrt(((R1X + R1W/2) - (CentroX))^2 + ((R1Y + R1H/2) - (CentroY))^2), {DistanzaRoom1 =< 187.0}, 
DistanzaRoom2 is sqrt(((R2X + R2W/2) - (CentroX))^2 + ((R2Y + R2H/2) - (CentroY))^2), {DistanzaRoom2 =< 187.0}, 
DistanzaRoom3 is sqrt(((R3X + R3W/2) - (CentroX))^2 + ((R3Y + R3H/2) - (CentroY))^2), {DistanzaRoom3 =< 187.0}, !


generateBathroom1(ZeroX, ZeroY, RoomWidth, RoomHeight, SI0X, SI0Y, SI0W, SI0H) :- 
repeat, 
Rwidthbound is RoomWidth + ZeroX, Rheightbound is RoomHeight + ZeroY, random(21.25, 29.75, SI0H), 
{SI0X + SI0W = Rwidthbound, SI0W = SI0H * (2/3)}, SIHSUB0 is Rheightbound - SI0H, random(ZeroY, SIHSUB0, SI0Y), 
{(322.80249596391894 =< SI0X ; SI0X + SI0W =< 297.30249596391894) ;
(230.6347694752165 =< SI0Y ; SI0Y + SI0H =< 209.3847694752165)}, !

generateBedroom0(ZeroX, ZeroY, RoomWidth, RoomHeight, B0X, B0Y, B0W, B0H, BS0X, BS0Y, BS0W, BS0H, W0X, W0Y, W0W, W0H) :- 
repeat, 
Rwidthbound is RoomWidth + ZeroX, Rheightbound is RoomHeight + ZeroY, {B0X + B0W =< Rwidthbound, B0Y + B0H =< Rheightbound}, 
{BS0X >= ZeroX, BS0Y >= ZeroY, BS0X + BS0W =< Rwidthbound, BS0Y + BS0H =< Rheightbound}, 
random(17.0, 25.5, B0W), {B0H = B0W + 25.5}, {B0X = ZeroX}, random(ZeroY, Rheightbound, B0Y), 
{BS0X = ZeroX, BS0Y = B0Y + B0H}, random(12.75, 17.0, BS0W), 
{BS0H = BS0W}, {W0X + W0W =< Rwidthbound, W0Y + W0H =< Rheightbound}, 
random(25.5, 63.75, W0W), random(12.75, 17.0, W0H), 
{W0Y = ZeroY}, random(ZeroX, Rwidthbound, W0X), 
{(345.14566136064684 =< B0X ; B0X + B0W =< 323.89566136064684) ; 
(145.88046154083122 =< B0Y ; B0Y + B0H =< 120.38046154083122)}, 
{(345.14566136064684 =< BS0X ; BS0X + BS0W =< 323.89566136064684) ; 
(145.88046154083122 =< BS0Y ; BS0Y + BS0H =< 120.38046154083122)}, 
{(345.14566136064684 =< W0X ; W0X + W0W =< 323.89566136064684) ; (145.88046154083122 =< W0Y ; W0Y + W0H =< 120.38046154083122)}, 
{(W0X + W0W =< B0X ; B0X + B0W =< W0X) ; (W0Y + W0H =< B0Y ; B0Y + B0H =< W0Y)}, 
{(BS0X + BS0W =< W0X ; W0X + W0W =< BS0X) ; (BS0Y + BS0H =< W0Y ; W0Y + W0H =< BS0Y)}, !

generateKitchen2(ZeroX, ZeroY, RoomWidth, RoomHeight, D0X, D0Y, D0W, D0H, D1X, D1Y, D1W, D1H, D2X, D2Y, D2W, D2H) :- 
repeat, 
Rwidthbound is RoomWidth + ZeroX, Rheightbound is RoomHeight + ZeroY, 
random(12.75, 15.3, DeskSize), D0X is ZeroX, {D0Y = Rheightbound - D0H}, 
D0H is DeskSize, DeskInfBound0 is RoomWidth * (7/10) - DeskSize, DeskSupBound0 is RoomWidth - DeskSize, 
random(DeskInfBound0, DeskSupBound0, D0W), 
{D1Y = Rheightbound - D1H}, D1W is DeskSize, DeskInfBound1 is RoomHeight * (7/10) - DeskSize, DeskSupBound1 is RoomHeight - DeskSize, 
random(DeskInfBound1, DeskSupBound1, D1H), D1X is Rwidthbound - DeskSize, D2X is ZeroX, {D2Y = Rheightbound - DeskSize - D2H}, 
D2W is DeskSize, DeskInfBound2 is RoomHeight * (7/10) - 2*DeskSize, DeskSupBound2 is RoomHeight - 2*DeskSize,  
random(DeskInfBound2, DeskSupBound2, D2H), random(5.949999999999999, 8.5, ChairSize), !

generateHall3(ZeroX, ZeroY, RoomWidth, RoomHeight, SO0X, SO0Y, SO0W, SO0H, SO1X, SO1Y, SO1W, SO1H, CB0X, CB0Y, CB0W, CB0H, CB1X, CB1Y, CB1W, CB1H) :- 
repeat, 
Rwidthbound is RoomWidth + ZeroX, Rheightbound is RoomHeight + ZeroY, 
random(5.949999999999999, 8.5, ChairSize), {CB0X + CB0W =< Rwidthbound, 
CB0Y + CB0H =< Rheightbound}, {CB1X + CB1W =< Rwidthbound, CB1Y + CB1H =< Rheightbound}, random(25.5, 102.0, CB0W), 
random(12.75, 17.0, CB0H), 
{CB0Y + CB0H = Rheightbound}, 
random(ZeroX, Rwidthbound, CB0X), random(12.75, 17.0, CB1W), 
random(25.5, 102.0, CB1H), {CB1X = ZeroX}, random(ZeroY, Rheightbound, CB1Y), 
random(21.25, 25.5, SO0H), random(51.0, 68.0, SO0W), SO0HSUB is ZeroY + RoomHeight/2 - SO0H, 
random(ZeroY, SO0HSUB, SO0Y), SO0WSUB is Rwidthbound - SO0W, random(ZeroX, SO0WSUB, SO0X), 
random(21.25, 25.5, SO1W), random(51.0, 68.0, SO1H), SO1HSUB is Rheightbound - SO1H, 
random(ZeroY, SO1HSUB, SO1Y), SO1WSUB is ZeroX + RoomWidth/2 - SO1W, 
random(ZeroX, SO1WSUB, SO1X), {(CB0X + CB0W =< SO0X - 8.5; SO0X + SO0W =< CB0X - 8.5) ; 
(CB0Y + CB0H =< SO0Y - 8.5 ; SO0Y + SO0H =< CB0Y - 8.5)}, 
{(CB1X + CB1W =< SO0X - 8.5; SO0X + SO0W =< CB1X - 8.5) ; 
(CB1Y + CB1H =< SO0Y - 8.5 ; SO0Y + SO0H =< CB1Y - 8.5)}, 
{(CB0X + CB0W =< SO1X - 8.5; SO1X + SO1W =< CB0X - 8.5) ; 
(CB0Y + CB0H =< SO1Y - 8.5 ; SO1Y + SO1H =< CB0Y - 8.5)}, 
{(CB1X + CB1W =< SO1X - 8.5; SO1X + SO1W =< CB1X - 8.5) ; 
(CB1Y + CB1H =< SO1Y - 8.5 ; SO1Y + SO1H =< CB1Y - 8.5)}, 
{(235.86156202838654 =< SO0X ; SO0X + SO0W =< 210.36156202838654) ; 
(126.4559709968834 =< SO0Y ; SO0Y + SO0H =< 105.2059709968834)}, 
{(235.86156202838654 =< SO1X ; SO1X + SO1W =< 210.36156202838654) ; 
(126.4559709968834 =< SO1Y ; SO1Y + SO1H =< 105.2059709968834)}, 
{(CB0X + CB0W =< CB1X ; CB1X + CB1W =< CB0X) ; (CB0Y + CB0H =< CB1Y ; CB1Y + CB1H =< CB0Y)}, 
{(235.86156202838654 =< CB0X ; CB0X + CB0W =< 210.36156202838654) ; 
(126.4559709968834 =< CB0Y ; CB0Y + CB0H =< 105.2059709968834)}, 
{(235.86156202838654 =< CB1X ; CB1X + CB1W =< 210.36156202838654) ; 
(126.4559709968834 =< CB1Y ; CB1Y + CB1H =< 105.2059709968834)}, {(SO0X + SO0W =< SO1X - 8.5; SO1X + SO1W =< SO0X - 8.5) ; 
(SO0Y + SO0H =< SO1Y - 8.5; SO1Y + SO1H =< SO0Y - 8.5)}, !
