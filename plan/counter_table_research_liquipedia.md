# Counter Table Research — Liquipedia (LotV pages), pulled 2026-10-10
Source API: liquipedia.net/starcraft2/api.php (gzip required), pages: <Unit>_(Legacy_of_the_Void).
Citations reference the same "# LP:" format as the existing COUNTER_TABLE in bot/combat/target_scoring.py.

## ADEPT (Gateway, Light, Biological; built today by PiG_Bot)
- LP: "strong against early Zealots and other light units"
- LP: "Adepts have no anti-air attack" (Muta/mass air harass kills them)
- LP: PvZ: Roaches out-tank; snowball vs bio in PvT "when paired with Phoenix"
- Pairs: (ADEPT, MARINE) +6.0 [light, early numbers]; (ADEPT, ZERGLING) +6.0 [LP: strong vs light/Zealots]; (ADEPT, ROACH) -6.0 [armored, out-tanks]; (ADEPT, MUTALISK) -6.0 [no anti-air]

## SENTRY (Gateway, Light, Mechanical; built today; NO direct attack in LotV)
- LP: Forcefields "guard the ramp and force favorable engagements" (alternative defense vs Zealots)
- LP (Guardian Shield): "effective against any low damage, high rate of attack type units (such as Marines)"
- Pairs: (SENTRY, MARINE) +3.0 [GS reduces marine damage; support role]; (SENTRY, ZERGLING) +3.0; (SENTRY, ZEALOT) +4.0 [LP: FF defense vs Zealots]
- NOTE: support/utility unit; scores stay small (it enables, it doesn't kill)

## DARK TEMPLAR (Dark Shrine, Light, Psionic, Biological)
- LP: "even a small number of DT can decimate a large number of workers very quickly" if undetected
- Pairs: (DARKTEMPLAR, MARINE) +5.0 [45-dmg melee shreds light if no detection]
- NOTE: detection-dependent; worker-line value not modeled in this table (workers excluded)

## PHOENIX (Stargate, Air, Light, Mechanical)
- LP: "great hit-and-run"; "effective at disabling caster units or destroying them outright. Most casters have light armor"; lifts high-threat units (Siege Tanks) — Graviton Beam per LP infobox
- Pairs: (PHOENIX, MUTALISK) +8.0 [classic Muta counter]; (PHOENIX, BANSHEE) +6.0 [light air, melts]; (PHOENIX, MEDIVAC) +6.0 [lifts/heal-kill]; (PHOENIX, VIPER) +6.0 [air, disables]; (PHOENIX, ORACLE) +5.0 [mirror: out-fights caster]; (PHOENIX, VIKINGFIGHTER) -8.0 [armored air, out-ranges 9>5, LP-tier hard counter]

## VOID RAY (Stargate, Air, Armored, Mechanical)
- LP: "versus an opponent that gets a lot of mechanical units like the Thor or the Tank, a few Void Rays can focus down the armored units very fast" (ramp damage)
- LP: drop defense — "if a boosted Medivac passes right by a charged Void Ray it will barely get through"
- LP: "Stalkers can deal with Void Rays easily because of the ease of making large amounts" ; VR vs Tempest: "Void Rays are extremely strong against Tempest (bonus damage against armored)"; PvP post-3.8: Carriers "easily countered by Stalkers + Sentries, Tempests, and Void Rays"
- Pairs: (VOIDRAY, SIEGETANK) +4.0; (VOIDRAY, THOR) +5.0 [LP: focus down armored]; (VOIDRAY, BATTLECRUISER) +8.0 [ramp on massive]; (VOIDRAY, TEMPEST) +8.0 [LP explicit]; (VOIDRAY, CARRIER) +6.0 [LP explicit]; (VOIDRAY, MEDIVAC) +5.0 [drop deterrent]; (VOIDRAY, MARINE) -6.0 [mass small arms]; (VOIDRAY, STALKER) -8.0 [LP explicit mirror]; (VOIDRAY, HYDRALISK) -6.0 [mass ranged]

## TEMPEST (Fleet Beacon, Air, Armored, Massive)
- LP: "Tempests can be used with late game army to snipe enemy Colossi and forcing the enemy to retreat"; "Void Rays are extremely strong against Tempest"
- Pairs: (TEMPEST, COLOSSUS) +8.0 [LP: snipe role, super-long range]; (TEMPEST, CARRIER) +7.0 [range war on big air]; (TEMPEST, BATTLECRUISER) +8.0 [bonus vs massive]; (VOIDRAY counter is a LOSS:) (TEMPEST, VOIDRAY) -6.0 [LP explicit]; (TEMPEST, MUTALISK) -4.0 [swarm bypasses range]

## CARRIER (Fleet Beacon, Air, Armored, Massive)
- LP: "extremely powerful versus stationary defenses and AI controlled units, since the AI will target the individual Interceptors first, not the Carriers themselves" (Patch 3.8.0 made Carriers "the dominant late game unit choice" in PvP)
- LP: Vikings: "killed in 2 volleys by Carriers... less than a third of the cost... the only match-up where the Carrier will lose the resource war"
- LP: PvP counters: "Stalkers + Sentries, Tempests, and Void Rays"; Zerg: Corruptors (same family as Colossus-counter line)
- Pairs: (CARRIER, PHOTONCANNON) +6.0 [vs static D, range outside cannon]; (CARRIER, SPORECRAWLER) +7.0; (CARRIER, SPINECRAWLER) +7.0; (CARRIER, VIKINGFIGHTER) -6.0 [LP resource war]; (CARRIER, CORRUPTOR) -5.0; (CARRIER, VOIDRAY) -5.0 [LP explicit PvP]

## ORACLE (Stargate, Air, Armored, Mechanical, Psionic) — EXCLUDED FROM TABLE BY DESIGN
- LP: "flying harassment and support caster"; PvZ meta harass "forces a response... Spore Crawlers and Queens"; "fairly expensive and fragile"
- Pulsar Beam: ground-only; vs ARMY units it has no matchup advantage — its value is economic harass + Revelation (anti-stealth intel) + Stasis Ward (traps)
- No pairs: consistent with the table's own rule "Only add entries where there's a real matchup advantage."

## MOTHERSHIP (Nexus ult, Air, Armored, Massive, Psionic) — EXCLUDED FROM TABLE BY DESIGN
- LP: "the primary use of the Mothership is its powerful support abilities" (Cloaking Field, Mass Recall, Time Warp); "Mothership play is usually ineffective against Terran"; niche support in Tempest-vs-BC battles
- No pairs: support ult; not a counter relationship vs any unit.
