import math
from bisect import bisect_left

from models.player import (
	Advertisement,
	Candidate,
	DateContext,
	DateOutcome,
	Decision,
	GameContext,
	PlayerSnapshot,
	RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import VALUES, Profile, decrement
from players.player_1.coord import Coordinator, is_signed, signature

from . import dates, robust
from .dpbar import DPBar

# No fixed positive floor; the declining acceptance bar still applies.
FLOOR_SHARE = 0.0
# The bar starts at this percentile of the first-date scores we have seen ...
BAR_PERCENTILE = 0.6
# ... or at this share of the bonus until we have MIN_SAMPLES of them.
BAR_START_SHARE = 0.7
MIN_SAMPLES = 5
# In the last share of rounds, accept any partner whose score is above 0.
ENDGAME_FRACTION = 0.2
# Act as in the endgame from round 1 when the game has at most URGENT_ROUNDS rounds, or
# when a random partner leaves us at or below 0 with probability >= URGENT_RISK (0 = off).
URGENT_ROUNDS = 10
URGENT_RISK = 0.0
# Untrusted partners become acceptable once this share of rounds is gone ...
RELAX_FRACTION = 0.5
# ... or when we know fewer than this many trusted partners above the bar.
MIN_TRUSTED_OPTIONS = 2
# A lie that costs us at most this share of the bonus is forgiven (endgame: the larger one).
LIE_TOLERANCE_SHARE = 0.1
LIE_TOLERANCE_ENDGAME_SHARE = 0.2
# Propose only if this partner is within this share of the bonus of our best other option.
BEST_SLACK_SHARE = 0.1
# A proposal from them counts as a signal for this many rounds.
WILLING_ROUNDS = 2
# Stop counting someone as an option after this many ignored proposals.
MAX_IGNORED = 3
# Lower the bar as the market empties, not only as rounds pass.
MARKET_CLOCK = True
# How the bar is set once we have MIN_SAMPLES first-date scores:
#   'percentile'  - BAR_PERCENTILE of those scores, scaled by time / market left.
#   'reservation' - the score we can still expect to beat with the chances left
#                   (unmet people, two rounds each, COMMIT_RATE of them willing).
BAR_MODE = 'percentile'
COMMIT_RATE = 0.5
# Measure COMMIT_RATE as we go: of the people we wanted after a first date, the share who
# came back (another date or a proposal). Starts at COMMIT_RATE, worth PRIOR_WEIGHT dates.
ADAPTIVE_COMMIT = False
PRIOR_WEIGHT = 4
# Advertisement: 'best' shows our best monotonic trait, 'appealing' the trait most
# people would like (high monotonic, or near the middle for regular traits).
AD_OWN = 'appealing'
# 'extreme' asks for our most demanding preference; 'lenient' our easiest one.
AD_SEEK = 'extreme'
# Share of never-dated people to list each round, best ad fit first (1 = all of them).
FRESH_SHARE = 1.0
# Only trim that list during this first share of the rounds; list everyone after.
TRIM_UNTIL = 1.0
# If fewer than this share of our first dates score above 0, accept any positive
# match straight away, as in the endgame. 0 turns it off.
SCARCE_SHARE = 0.0
# The same rule in big markets, where it measured +3 points of positive rate at B=50.
BIG_SCARCE_SHARE = 0.35
# Before proposing, re-score the partner as if they used the consistent lies other
# groups are known to use (shifting the trait our ad seeks by up to 2, or masking
# extreme values), and only propose if even that worst case is positive.
SAFETY_CHECK = False
# Only compare against people who showed interest back (a second date or a proposal),
# not everyone we ever dated. In big markets the best stranger rarely wants us too.
ENGAGED_ONLY = False
# In markets with at least this many people, use the reservation bar and compare only
# against engaged partners. Both help with hundreds of options and hurt with few.
BIG_MARKET = 100
# After every PAUSE_AFTER ignored proposals, stop asking that person out for this many
# rounds (0 = never). Big markets only: slow committers otherwise use up our dates.
PAUSE_AFTER = 2
BIG_PAUSE_ROUNDS = 0
# List people who proposed to us first, even below the bar, while still above the floor.
LIST_PROPOSERS = False
# BAR_MODE = 'dp' (see dpbar.py) replaces the bar, the endgame, scarce and big-market
# modes and the best-other check with one dynamic-programming reservation value.
# The DP_* values below are the tuned ones; they only matter once BAR_MODE = 'dp'.
# Utility of a match: happiness + DP_LAMBDA * bonus if positive.
DP_LAMBDA = 2.0
# Use the DP bar in markets at least this big even when BAR_MODE is not 'dp'.
DP_MIN_PEOPLE = 500
# Score distribution from this many recent first dates (0 = all of them).
DP_WINDOW = 50
# Fall back on the partners we know (by p_commit) once fresh chances run out.
DP_POOL = True
# Someone who just proposed to us is accepted this share of the bonus below the bar.
DP_DELTA_SHARE = 0.1
# Ask for another date with anyone acceptable ('accept'), or only when that date is worth
# at least a fresh one ('worth': p_commit * (U - V_{k-1}) >= V_k - V_{k-1}).
DP_AGAIN = 'accept'
# List a known partner if that date is worth a fresh one ('worth') or if acceptable ('bar').
DP_LIST = 'worth'
# Rounds to stop asking out someone after every PAUSE_AFTER of our proposals they ignored.
DP_PAUSE_ROUNDS = 5
# ... but only while there are strangers left to meet instead.
DP_PAUSE_WHILE_FRESH = True
# Prior chances that a partner we want asks again (stage 1), then proposes (stage 2).
DP_Q1_PRIOR = 0.5
DP_Q2_PRIOR = 0.7
# Prior for how fast singles leave: this many e-folds over the game (0 = measured only),
# worth DP_RHO_WEIGHT rounds of measurement.
DP_RHO_FOLDS = 2.3
DP_RHO_WEIGHT = 10
# Use the old start-up bar (BAR_START_SHARE) until we have this many first dates.
DP_MIN_SAMPLES = 0
# Until we have this many first dates, fill the score distribution with synthetic partners
# (traits resampled from what we've seen, smoothed with a uniform prior). 0 = off.
DP_PRIOR_N = 10
# Ramp the positivity bonus in over the first this share of the bonus (0 = a step at 0).
DP_RAMP_SHARE = 0.0
# Count at most this many mutual agains as evidence that someone will commit.
DP_MUTUAL_CAP = 1000
# Count a first date as a real chance only if they would be happy with us too (judged
# from their stated ideal); known partners who wouldn't are not expected to commit.
DP_MUTUAL_X = True
# At most this many future chances (0 = no cap); each costs DP_ROUNDS_PER_CHANCE rounds.
DP_K_CAP = 0
DP_ROUNDS_PER_CHANCE = 1.0
# A first-date "score" that never clears any bar: they would not want us.
NO_CHANCE = -1e9
# High-bonus mode. When a marriage can (almost) never be non-positive for us, the
# positive rate only depends on getting married at all, and groups are ranked on mean
# happiness. We estimate that risk at the start: 0 if the bonus beats our worst possible
# decrement plus half of theirs (as Player 9 does), else P(score <= 0) with uniform traits
# and ideals. At or below HB_MAX_RISK, 'dp' uses the DP bar (which waits for a good match
# instead of taking the first proposer) at every market size from HB_MIN_PEOPLE, with
# HB_LAMBDA in place of DP_LAMBDA: plain mean happiness, where being single already costs
# about B. 'off' changes nothing.
# Measured (Nov 11, B300 cells vs flags off, positive rate unchanged at 100%): N200 t60
# +6.5 [+3.8, +9.1], N200 t200 +10.5 [+8.8, +12.0], N1000 t350 +2.2 [+1.3, +3.2].
# Worse: DP_LAMBDA's 2 (t60 -0.3), 0.5 (+5.8 vs +10.9 at 0), markets of 20-30 people
# (-7), a secretary-style quantile bar in the percentile machinery (+3.4 t60, +0.9 t200),
# lambda scaled with the risk, other rho priors, and trimming the stranger list.
HB_MODE = 'dp'
HB_MAX_RISK = 0.05
HB_LAMBDA = 0.0
HB_MIN_PEOPLE = 50
# Coordinate with our own copies (coord.py): sign the ad's seek half, date every copy
# once on a round-robin schedule, then all copies solve the same matching and marry it.
COORDINATE = True
# Targeted lie for when we are short of options: 'off', 'nudge' (move the trait the
# partner's ad seeks up to PERSUADE_STEP toward their wish) or 'mirror' (also move our
# other traits toward the ideal they stated on an earlier date). Never toward copies.
# Monotonic traits only move up. The story stays fixed per partner once shown.
PERSUADE = 'mirror'
PERSUADE_STEP = 3
# Only in markets up to this size. Big markets measured +0.19pp positive (1st pooled)
# for +0.11pp bad marriages, accepted on Nov 10.
PERSUADE_MAX_PEOPLE = 1_000_000
# "Short of options": fewer than this share of first dates scored above 0 ...
PERSUADE_SHARE = 1.01
# The same threshold in markets of BIG_MARKET people or more.
BIG_PERSUADE_SHARE = 0.2
# ... or the endgame (when PERSUADE_ENDGAME is on).
PERSUADE_ENDGAME = True
# Crowd-pleaser first-date profile. On date 1 we only know their ad, so the rest of our
# story is set to please a typical partner: monotonic traits pulled up toward
# CROWD_MONO, regular traits toward the mean stated ideal seen so far (a uniform prior
# centred on 5, worth CROWD_PRIOR reports). CROWD_BLEND is how far to move (0 = true,
# 1 = all the way), at most CROWD_STEP per trait. The trait our ad shows stays true, their
# ad's seek trait is still nudged toward their wish, and the story stays fixed per
# partner. Never toward our copies. 'off', 'needy' (whenever the persuasion lie is on)
# or 'all' (every first date, also before we have MIN_SAMPLES of them).
# Measured (Nov 11, vs off; mix tier = N20 round-robin market, 200 seeds/cell):
#   'all' blend 0.3/0.5/0.6/1.0: positive +3.2/+3.8/+3.8/+2.9pp; 'needy' 0.6: +2.6pp.
#   'all' 0.5 + CROWD_FLOOR_SHARE 0.1: +3.80% [+3.15, +4.43], mean/B +0.006, bad -0.02%
#   [-0.17, +0.13]. Held-out tier A ('all' 0.6): +1.37% [+0.77, +1.96], kind=mix +4.9pp.
#   CROWD_NUDGE_GATE trades about 0.4-0.6pp of that for about -0.07pp bad marriages.
CROWD = 'all'
CROWD_BLEND = 0.5
# Blend for the regular traits alone (below 0: CROWD_BLEND).
CROWD_REG_BLEND = -1.0
CROWD_STEP = 10
CROWD_MONO = 10
CROWD_PRIOR = 4.0
# Only in markets up to this size. Held-out big markets (N50-N500, DP bar) measured
# mean/B -0.014 [-0.022, -0.007] and bad +0.44% with it on everywhere.
CROWD_MAX_PEOPLE = 49
# While the crowd-pleaser is on, accept nobody scoring at or below this share of the
# bonus (more people want us, so we can afford to skip the near-zero, lie-prone matches).
CROWD_FLOOR_SHARE = 0.1
# While the crowd-pleaser is on, also use ROBUST_NUDGE_GATE (more people want us, so we
# can afford to skip matches a 2-step nudge of our seek trait would make non-positive).
CROWD_NUDGE_GATE = False
# Hard-to-match detection in round 1: from every candidate's ad (and a uniform prior on
# the traits it doesn't show), estimate how many people could make us happy. Below this
# many, act as in the endgame from the start: accept any positive match (0 = off).
SCARCE_OPTIONS = 0.0
SCARCE_DRAWS = 24
# Only in markets this small (larger ones keep the plain ad: there, matching among a
# tenth of the seats measured worse than searching the whole market) ...
COORD_MAX_PEOPLE = 99
# ... holding this many copies, and this share of the seats ...
COORD_MIN_COPIES = 4
COORD_MAX_COPIES = 60
COORD_MIN_MARKET_SHARE = 0.25
# ... and the game has this many rounds beyond the exploration schedule.
COORD_ROUND_SLACK = 10
# Extra rounds to make up a missed exploration date before giving up on coordinating.
COORD_RETRY_ROUNDS = 3
# Rounds (or unanswered dates) to wait for the assigned partner before searching normally.
COORD_PATIENCE = 3
# A copy pair is matched only if this many of its two seats come out positive ...
COORD_MIN_POSITIVE = 2
# ... and both sides get at least this share of the bonus.
COORD_MIN_SHARE = 0.0
# Lies other groups use that leave no trace on a date (see robust.py): any of 'mask'
# (regular traits clamped into [2, 8]), 'nudge' (the trait our ad seeks moved toward our
# wish) and 'blend' (a stated ideal pulled toward us). Empty turns all of this off.
# Measured best (bench, Nov 10): ROBUST_LIES=('mask',), ROBUST_GATE='always',
# ROBUST_SUSPECT=0.95, ROBUST_PRICE_FLOOR=0.95, the rest at their defaults. It cut our
# bad marriages under U6 from 0.35% to 0.08% (N=100 U6: 2.08% to 0.21%) at no measurable
# cost in positive rate. 'nudge' and 'blend' cannot tell liars from honest people (their
# posterior stays at the prior), so they only add pessimism and cost positive rate.
ROBUST_LIES: tuple[str, ...] = ('mask',)
# Prior probability of each lie, and the posterior at which we call someone a suspect.
ROBUST_PRIOR = 0.1
ROBUST_SUSPECT = 0.95
# Subtract the posterior-weighted expected cost of those lies from a partner's score,
# counting only lies at least ROBUST_PRICE_FLOOR likely (where traits at 2 and 8 are
# common anyway, weak suspicion of everyone only costs us matches) ...
ROBUST_PRICE = True
ROBUST_PRICE_FLOOR = 0.95
# ... and count it as lie cost, so suspects wait until untrusted partners are allowed.
ROBUST_TRUST = False
# Propose to a suspect only if their score stays above ROBUST_GATE_MARGIN (share of the
# bonus) when every suspected lie hides the worst value it could: 'off', 'always', or
# only in the 'endgame'.
ROBUST_GATE = 'always'
ROBUST_GATE_MARGIN = 0.0
# Whether the gate also declines a second date after a first one. Off, a suspect still
# gets the second date: by then we have more first dates, and their posterior is firmer.
ROBUST_GATE_AGAIN = True
# Suspicion needs a histogram: with fewer first dates than this, no lie is suspected.
ROBUST_MIN_REPORTS = 0
# Before the endgame, propose to anyone only if their score stays above 0 when the trait
# our ad seeks was nudged 2 steps toward our wish (Player 8's lie, which no one can tell
# apart from honesty). Bad marriages to nudgers happen early, at small positive scores.
ROBUST_NUDGE_GATE = False
# Whom to list (see dates.py). 'ad' ranks strangers by the two advertised traits alone
# (the original behaviour, and the default). 'ev' ranks them by our expected happiness,
# filling the unadvertised traits from the learned distributions. 'mutual' ranks them by
# the chance that both of us clear our bars, plus DATES_QUALITY_W x our expected share.
# 'ad2' is 'ad' ranking with the known-partner handling of dates.py.
# Measured best: DATE_MODE='ev' with DATES_FRESH_SHARE=0.15 (full tier vs flags off:
# N200 t60/t200 positive +1.33% [+0.76, +1.95], t60 B50 +6.4%; A mixed +0.49%, mean/B
# +2.3%; N1000 t120 +0.38%). Listing fewer strangers is what helps; 'mutual' and the
# known-partner caps measured no better.
DATE_MODE = 'ev'
# Uniform prior weight per dimension, and how much the sample must grow before the
# distributions are rebuilt.
DATES_PRIOR = 2.0
DATES_REFRESH = 0.2
# 'mutual' thresholds: ours is max(bar, this share of B); theirs is this share of B,
# scaled by the share of rounds left.
DATES_US_MIN_SHARE = 0.0
DATES_THEM_SHARE = 0.2
DATES_QUALITY_W = 0.5
# Also list people who proposed to us within WILLING_ROUNDS, even below the bar.
DATES_LIST_WILLING = True
# List at most this many known partners (0 = all of them), and rank people who showed
# interest back first.
DATES_KNOWN_TOP = 0
DATES_RECIP_FIRST = False
# Strangers listed, as a share of the list limit (until DATES_TRIM_UNTIL of the rounds),
# and while we have a strong engaged option (above the bar and showing interest back).
DATES_FRESH_SHARE = 0.15
DATES_TRIM_UNTIL = 1.0
DATES_ENGAGED_FRESH_SHARE = 1.0
DATES_MIN_FRESH = 2
# If fewer than this share of our first DATES_SCARCE_MIN dates or more scored above 0,
# rank strangers by the chance that both of us would be positive (0 = off).
DATES_SCARCE = 0.0
DATES_SCARCE_MIN = 5


def _date_params() -> dict:
	"""The listing constants, read at call time so benchmarks can override them."""
	return {
		'MODE': DATE_MODE,
		'PRIOR': DATES_PRIOR,
		'REFRESH_GROWTH': DATES_REFRESH,
		'US_MIN_SHARE': DATES_US_MIN_SHARE,
		'THEM_SHARE': DATES_THEM_SHARE,
		'QUALITY_W': DATES_QUALITY_W,
		'LIST_WILLING': DATES_LIST_WILLING,
		'WILLING_ROUNDS': WILLING_ROUNDS,
		'KNOWN_TOP': DATES_KNOWN_TOP,
		'RECIP_FIRST': DATES_RECIP_FIRST,
		'FRESH_SHARE': DATES_FRESH_SHARE,
		'TRIM_UNTIL': DATES_TRIM_UNTIL,
		'ENGAGED_FRESH_SHARE': DATES_ENGAGED_FRESH_SHARE,
		'MIN_FRESH': DATES_MIN_FRESH,
		'SCARCE': DATES_SCARCE,
		'SCARCE_MIN': DATES_SCARCE_MIN,
	}


class Player1(BasePlayer):
	"""Trust but verify, relax over time.

	We are always honest ourselves. Every partner is scored with the handout's
	formula on what they showed us, after "repricing" anything that looks like a
	lie at its most pessimistic plausible value:
	- a trait that differs between their ad and their behaviour, or between dates,
	- a stated ideal that is an exact copy of our profile (flattery), together with
	the trait our ad asked for.
	The size of that correction is the lie's cost. Small lies are forgiven; partners
	with bigger lies are only considered later in the game.

	We propose only to partners above a personal bar that falls as rounds run out,
	and only if nobody we have dated and could still marry looks clearly better.
	In the endgame, anyone whose repriced score is above 0 is acceptable.
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.floor = max(0.0, FLOOR_SHARE * self.bonus)
		self.coordinating = COORDINATE and self.people <= COORD_MAX_PEOPLE
		self.my_ad = self._make_ad()
		# partner -> distinct (projection, stated ideal) reports, first one first
		self.reports: dict[int, list[tuple[Profile, Profile]]] = {}
		self.ads: dict[int, Advertisement] = {}
		# partner -> (repriced score, lie cost)
		self.scores: dict[int, tuple[float, float]] = {}
		self.blocked: set[int] = set()
		self.safe: dict[int, float] = {}  # partner -> worst-case score under known lies
		self.samples: list[float] = []
		# What typical people look like, from first dates without red flags.
		self.dim_values: list[list[int]] = [[] for _ in range(self.dims)]
		self.their_costs: list[int] = []
		self.proposed_to_me: dict[int, int] = {}
		self.ignored: dict[int, int] = {}
		self.proposed_to: set[int] = set()
		self.met: dict[int, int] = {}  # partner -> dates we've had
		self.wanted: set[int] = set()  # people we wanted after a first date
		self.returned: set[int] = set()  # ... who then came back to us
		self.dp: DPBar | None = None
		self.hb = (
			HB_MODE == 'dp'
			and self.people >= HB_MIN_PEOPLE
			and self._positive_risk() <= HB_MAX_RISK
		)
		self.urgent = bool(URGENT_ROUNDS and self.rounds <= URGENT_ROUNDS) or bool(
			URGENT_RISK and self._positive_risk() >= URGENT_RISK
		)
		if BAR_MODE == 'dp' or self.people >= DP_MIN_PEOPLE or self.hb:
			# One continuous rule for every market size: no big-market switch.
			self.dp = DPBar(
				self.bonus,
				lam=HB_LAMBDA if self.hb else DP_LAMBDA,
				window=DP_WINDOW,
				ramp_share=DP_RAMP_SHARE,
				mutual_cap=DP_MUTUAL_CAP,
				q1_prior=DP_Q1_PRIOR,
				q2_prior=DP_Q2_PRIOR,
				rho_prior=DP_RHO_FOLDS / max(1, self.rounds),
				rho_weight=DP_RHO_WEIGHT if DP_RHO_FOLDS else 0,
			)
			if DP_PRIOR_N:
				self.dp.set_prior(self._synthetic_scores(), DP_PRIOR_N)
			self.big = False
			self.bar_mode = 'dp'
			self.engaged_only = ENGAGED_ONLY
			self.scarce_share = 0.0
			self.pause_rounds = DP_PAUSE_ROUNDS
		else:
			self.big = self.people >= BIG_MARKET
			self.bar_mode = 'reservation' if self.big else BAR_MODE
			self.engaged_only = ENGAGED_ONLY or self.big
			self.scarce_share = BIG_SCARCE_SHARE if self.big else SCARCE_SHARE
			self.pause_rounds = BIG_PAUSE_ROUNDS if self.big else 0
		self.paused_until: dict[int, int] = {}
		self.shown: dict[int, Profile] = {}  # partner -> what we projected to them
		self.mirrored: set[int] = set()
		self.crowd_base: dict[int, Profile] = {}  # partner -> crowd-pleaser story
		# Stated ideals of typical people, per dimension (for the crowd-pleaser profile).
		self.ideal_values: list[list[int]] = [[] for _ in range(self.dims)]
		self.options: float | None = None  # expected number of people positive for us
		self.available: set[int] = set()
		self.singles_left = 1.0  # share of the market still single, as of our last choice
		self.unmet = self.people - 1  # singles we have never dated, as of our last choice
		self.views: dict[int, float] = {}  # partner -> how happy they would be with us
		self.listed: tuple[int, set[int]] = (0, set())  # (round, who we listed)
		self.dated_in: set[int] = set()  # rounds in which we had a date
		self.coord = (
			Coordinator(
				self.index,
				self.profile,
				self.ideal,
				self.people,
				self.rounds,
				self.bonus,
				self.monotonic,
				self._ad_for,
				min_copies=COORD_MIN_COPIES,
				max_copies=COORD_MAX_COPIES,
				min_market_share=COORD_MIN_MARKET_SHARE,
				round_slack=COORD_ROUND_SLACK,
				retry_rounds=COORD_RETRY_ROUNDS,
				patience=COORD_PATIENCE,
				min_positive=COORD_MIN_POSITIVE,
				min_share=COORD_MIN_SHARE,
			)
			if self.coordinating
			else None
		)
		self.robust = (
			robust.LieModel(
				self,
				ROBUST_LIES,
				ROBUST_PRIOR,
				ROBUST_SUSPECT,
				ROBUST_PRICE_FLOOR,
				ROBUST_MIN_REPORTS,
			)
			if ROBUST_LIES
			else None
		)
		self.worst: dict[int, float] = {}  # partner -> score if every suspected lie is worst
		self.asked_again: dict[int, bool] = {}  # partner -> did we ask for another date
		self.their_again: dict[int, bool] = {}  # partner -> did they (when we asked)
		self.date_model = dates.DateModel(self.profile, self.ideal, self.monotonic, self.bonus)

	# ------------------------------------------------------------------ moves

	def advertise(self) -> Advertisement:
		return self.my_ad

	def project(self, date: DateContext) -> Profile:
		partner = date.partner
		# Mirroring needs their stated ideal, so it may change our story once, late on.
		remirror = (
			PERSUADE == 'mirror'
			and partner in self.reports
			and partner not in self.mirrored
			and self._endgame(date.rounds_left)
			and self.scores.get(partner, (0.0, 0.0))[0] > 0
		)
		if partner in self.shown and not remirror:
			return self.shown[partner]
		if remirror:
			self.mirrored.add(partner)
		if self._needy(date.rounds_left) or self._crowd_on(needy=False):
			shown = self._persuasive(date)
		else:
			shown = None
		shown = tuple(self.profile) if shown is None else shown
		self.shown[partner] = shown
		return shown

	def _needy(self, rounds_left: int) -> bool:
		if PERSUADE == 'off' or self.people > PERSUADE_MAX_PEOPLE:
			return False
		if (PERSUADE_ENDGAME and self._endgame(rounds_left)) or self._early_scarce():
			return True
		if len(self.samples) < MIN_SAMPLES:
			return False
		share = BIG_PERSUADE_SHARE if self.big else PERSUADE_SHARE
		return sum(1 for s in self.samples if s > 0) / len(self.samples) < share

	def _persuasive(self, date: DateContext) -> Profile | None:
		ad = date.partner_ad
		if is_signed(ad):
			return None  # possibly a copy: copies must see the truth
		shown = list(self.profile)
		keep = self.my_ad.own_dim  # our ad shows this trait; changing it would be caught

		def toward(dim: int, want: int) -> None:
			if dim == keep:
				return
			have = shown[dim]
			step = max(-PERSUADE_STEP, min(PERSUADE_STEP, want - have))
			if dim < self.monotonic:
				step = max(0, step)
			shown[dim] = max(0, min(10, have + step))

		if self._crowd_on(needy=True):
			# One crowd base per partner, so a late re-mirror starts from the same story.
			base = self.crowd_base.get(date.partner)
			if base is None:
				base = self.crowd_base[date.partner] = self._crowd_profile()
			shown = list(base)
			if ad.seek_dim < self.monotonic:
				nudged = self.profile[ad.seek_dim] + max(
					0, min(PERSUADE_STEP, ad.seek_value - self.profile[ad.seek_dim])
				)
				shown[ad.seek_dim] = max(shown[ad.seek_dim], min(10, nudged))
			else:
				shown[ad.seek_dim] = self.profile[ad.seek_dim]
				toward(ad.seek_dim, ad.seek_value)
		else:
			toward(ad.seek_dim, ad.seek_value)
		reports = self.reports.get(date.partner)
		if PERSUADE == 'mirror' and reports:
			wish = reports[-1][1]
			for dim in range(self.dims):
				if dim != ad.seek_dim:
					toward(dim, int(wish[dim]))
		return tuple(shown)

	def _crowd_on(self, needy: bool) -> bool:
		if self.people > CROWD_MAX_PEOPLE:
			return False
		return CROWD == 'all' or (CROWD == 'needy' and needy)

	def _crowd_floor(self) -> float:
		return CROWD_FLOOR_SHARE * self.bonus if self._crowd_on(needy=True) else 0.0

	def _crowd_profile(self) -> Profile:
		"""Our profile moved toward what a typical partner wants (our ad trait kept true)."""
		shown = list(self.profile)
		for dim in range(self.dims):
			if dim == self.my_ad.own_dim:
				continue
			have = self.profile[dim]
			if dim < self.monotonic:
				target = float(CROWD_MONO)
			else:
				seen = self.ideal_values[dim]
				target = (CROWD_PRIOR * 5.0 + sum(seen)) / (CROWD_PRIOR + len(seen))
			blend = CROWD_BLEND
			if dim >= self.monotonic and CROWD_REG_BLEND >= 0:
				blend = CROWD_REG_BLEND
			move = blend * (target - have)
			move = max(-CROWD_STEP, min(CROWD_STEP, move))
			value = int(math.floor(have + move + 0.5))
			if dim < self.monotonic:
				value = max(have, value)
			shown[dim] = max(0, min(10, value))
		return tuple(shown)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self.available = {c.index for c in candidates}
		if self.options is None and SCARCE_OPTIONS:
			self.options = self._estimate_options(candidates)
		self.singles_left = turn.singles / max(1, self.people)
		self.unmet = sum(1 for c in candidates if c.dates == 0)
		if self.dp is not None:
			return self._coord_choose(candidates, turn, self._dp_choose(candidates, turn))
		if DATE_MODE != 'ad':
			chosen = dates.choose(self, candidates, turn, _date_params())
			if self.coord is not None:
				override = self.coord.choose(candidates, turn.round)
				if override is not None:
					return override[: self.list_limit]
			return chosen
		bar = self._bar(turn.rounds_left)
		endgame = self._endgame(turn.rounds_left)
		relaxed = self._relaxed(turn.rounds_left, bar)

		floor = self._floor(turn.rounds_left)
		# People who ignore our proposals sort last, but stay listed: dropping them
		# measured worse, since slow committers (e.g. Player 5) often come round.
		proposers = []
		if LIST_PROPOSERS:
			proposers = [
				i
				for i in sorted(self.proposed_to_me, key=lambda i: -self.scores[i][0])
				if i in self.available
				and i not in self.blocked
				and self.scores[i][0] > floor
				and turn.round - self.proposed_to_me[i] <= WILLING_ROUNDS
			]
		known = sorted(
			(
				c.index
				for c in candidates
				if c.index in self.scores
				and c.index not in self.blocked
				and self.paused_until.get(c.index, 0) < turn.round
			),
			key=lambda i: (self._unresponsive(i), -self.scores[i][0], i),
		)
		good = [
			i for i in known if self.scores[i][0] >= bar and self._eligible(i, relaxed, endgame)
		]
		fresh = sorted(
			(c for c in candidates if c.index not in self.reports),
			key=lambda c: (self._ad_cost(c.ad), c.index),
		)
		# The engine treats our list as a set, so ranking only matters if we cut it:
		# list just the best-fitting share of strangers to spend dates on likely matches.
		if FRESH_SHARE < 1 and turn.round <= TRIM_UNTIL * self.rounds:
			fresh = fresh[: max(2, math.ceil(FRESH_SHARE * len(fresh)))]
		chosen = proposers + [i for i in good if i not in proposers] + [c.index for c in fresh]
		# Weaker known partners are a fallback once strangers run out or time does.
		if endgame or not fresh:
			chosen += [
				i
				for i in known
				if i not in good
				and self.scores[i][0] > floor
				and self._eligible(i, relaxed, endgame)
			]
		return self._coord_choose(candidates, turn, chosen)

	def _coord_choose(
		self, candidates: tuple[Candidate, ...], turn: RoundContext, chosen: list[int]
	) -> list[int]:
		if self.coord is not None:
			override = self.coord.choose(candidates, turn.round)
			if override is not None:
				return override[: self.list_limit]
		return chosen[: self.list_limit]

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		decision = self._decide(date, observed, stated_ideal)
		if self.coord is not None:
			override = self.coord.decide(date, observed, stated_ideal, decision)
			if override is not None:
				if not override.incompatible:
					self.blocked.discard(date.partner)
				if not override.propose:
					self.proposed_to.discard(date.partner)
				if self.dp is not None:
					self.dp.note_decision(date.partner, override.again, override.propose)
				return override
		return decision

	def _decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		partner = date.partner
		self.ads[partner] = date.partner_ad
		first_date = partner not in self.reports
		self.met[partner] = self.met.get(partner, 0) + 1
		reports = self.reports.setdefault(partner, [])
		if (observed, stated_ideal) not in reports:
			reports.append((observed, stated_ideal))

		if first_date and self.robust:
			self.robust.learn(partner, observed, date.partner_ad, stated_ideal)
		score, lie_cost = self._reprice(partner)
		if self.robust:
			expected, worst, _ = self.robust.assess(partner)
			if worst > 0:
				self.worst[partner] = score - worst
			else:
				self.worst.pop(partner, None)
			if ROBUST_PRICE:
				score -= expected
			if ROBUST_TRUST:
				lie_cost += max(0.0, expected)
		self.scores[partner] = (score, lie_cost)
		self.safe[partner] = self._worst_case(partner)
		if first_date:
			self.samples.append(score)
			self._learn(partner, observed, stated_ideal)
		if self.dp is not None:
			return self._dp_decide(date, score, first_date)

		rounds_left = date.rounds_left
		bar = self._bar(rounds_left)
		floor = self._floor(rounds_left)
		endgame = self._endgame(rounds_left)
		relaxed = self._relaxed(rounds_left, bar)

		# Block only clear negatives, and only while there is time to find others.
		if score <= 0 and not endgame:
			self.blocked.add(partner)
			return Decision(incompatible=True)

		eligible = self._eligible(partner, relaxed, endgame)
		willing = date.round - self.proposed_to_me.get(partner, -100) <= WILLING_ROUNDS
		best_other = self._best_other(partner, relaxed, endgame)

		if endgame or willing or self._scarce() or self._early_scarce():
			want = eligible and score > max(floor, self._crowd_floor())
		else:
			want = eligible and score >= bar and score >= best_other - BEST_SLACK_SHARE * self.bonus

		if SAFETY_CHECK and self.safe[partner] <= 0:
			want = False
		gated = self._robust_unsafe(partner, endgame)
		if gated and (ROBUST_GATE_AGAIN or date.date_number >= 2):
			want = False
		if first_date and want:
			self.wanted.add(partner)
		elif partner in self.wanted:
			self.returned.add(partner)
		propose = want and not gated and date.date_number >= 2
		if propose:
			self.proposed_to.add(partner)
		# Another date is how we verify and propose; keep it for people we'd marry.
		paused = self.paused_until.get(partner, 0) >= date.round
		self.asked_again[partner] = want and not paused
		return Decision(again=want and not paused, propose=propose)

	def on_outcome(self, outcome: DateOutcome) -> None:
		if self.coord is not None:
			self.coord.on_outcome(outcome)
		partner = outcome.partner
		if self.dp is not None:
			self.dp.note_outcome(
				partner, outcome.date_number, outcome.partner_proposed, outcome.again
			)
		if outcome.partner_proposed:
			self.proposed_to_me[partner] = outcome.round
			if partner in self.wanted:
				self.returned.add(partner)
		if partner in self.proposed_to and not outcome.married:
			self.ignored[partner] = self.ignored.get(partner, 0) + 1
			if self.pause_rounds and self.ignored[partner] % PAUSE_AFTER == 0:
				self.paused_until[partner] = outcome.round + self.pause_rounds
		self.proposed_to.discard(partner)
		if outcome.again:
			self.their_again[partner] = True
		elif self.asked_again.get(partner) and not outcome.married:
			self.their_again[partner] = False

	# ----------------------------------------------------- dynamic programming

	def _positive_risk(self) -> float:
		"""P(our score <= 0) for a partner with uniform traits and ideal (0 if impossible)."""
		mine_dims, their_dims = [], []
		for dim in range(self.dims):
			mine: dict[int, float] = {}
			theirs: dict[int, float] = {}
			for v in VALUES:
				c = self._gap(dim, self.ideal[dim], v)
				mine[c] = mine.get(c, 0.0) + 1 / len(VALUES)
				c = self._gap(dim, v, self.profile[dim])
				theirs[c] = theirs.get(c, 0.0) + 1 / len(VALUES)
			mine_dims.append(mine)
			their_dims.append(theirs)
		if self.bonus > sum(max(d) for d in mine_dims) + sum(max(d) for d in their_dims) / 2:
			return 0.0
		mine_total = dates._convolve(mine_dims)
		their_total = sorted(dates._convolve(their_dims).items())
		support = [t for t, _ in their_total]
		# tail[i] = P(T >= support[i]), a suffix sum over the sorted support.
		tail = [0.0] * (len(their_total) + 1)
		for i in range(len(their_total) - 1, -1, -1):
			tail[i] = tail[i + 1] + their_total[i][1]
		risk = 0.0
		for m, pm in mine_total.items():
			# score <= 0  <=>  T >= 2 (B - M)
			risk += pm * tail[bisect_left(support, 2 * (self.bonus - m))]
		return risk

	def _synthetic_scores(self, count: int = 100) -> list[float]:
		"""First-date scores of made-up partners, for before we have real ones."""
		pools = [self.dim_values[d] + list(VALUES) for d in range(self.dims)]
		values = list(VALUES)
		out = []
		for _ in range(count):
			theirs = tuple(self.rng.choice(pool) for pool in pools)
			if self.their_costs:
				cost = self.rng.choice(self.their_costs)
			else:
				wish = tuple(self.rng.choice(values) for _ in range(self.dims))
				cost = decrement(wish, self.profile, self.monotonic)
			mine = decrement(self.ideal, theirs, self.monotonic)
			score = self.bonus - mine - cost / 2
			if DP_MUTUAL_X and self.bonus - cost - mine / 2 <= 0:
				score = NO_CHANCE
			out.append(score)
		return out

	def _their_view(self, partner: int) -> float:
		"""How happy this partner would be with us, by their stated ideal."""
		reports = self.reports[partner]
		honest = [ideal for _, ideal in reports if tuple(ideal) != tuple(self.profile)]
		if honest:
			theirs = decrement(honest[-1], self.profile, self.monotonic)
		else:
			theirs = self._typical_their_cost()
		mine = decrement(self.ideal, reports[-1][0], self.monotonic)
		return self.bonus - theirs - mine / 2

	def _p_commit(self, partner: int) -> float:
		doubtful = partner not in self.proposed_to_me and self.views.get(partner, 1.0) <= 0
		if DP_MUTUAL_X and doubtful:
			return 0.0
		return self.dp.p_commit(partner)

	def _dp_paused(self, partner: int, rnd: int) -> bool:
		"""Cooling off a slow committer only pays while strangers are left to meet."""
		if DP_PAUSE_WHILE_FRESH and not self.unmet:
			return False
		return self.paused_until.get(partner, 0) >= rnd

	def _dp_value(self, rnd: int, rounds_left: int, exclude: int | None) -> tuple[float, float]:
		"""(V_k, V_{k-1}): what carrying on is worth, without partner ``exclude``."""
		dp = self.dp
		if len(self.samples) < DP_MIN_SAMPLES:
			left = max(0.0, (rounds_left - 1) / max(1, self.rounds - 1))
			v = dp.utility(BAR_START_SHARE * self.bonus * min(left, self.singles_left))
			return v, v
		v0 = 0.0
		if DP_POOL:
			endgame = self._endgame(rounds_left)
			late = rounds_left <= (1 - RELAX_FRACTION) * self.rounds
			options = [
				(dp.utility(score), self._p_commit(i))
				for i, (score, _) in self.scores.items()
				if i != exclude
				and score > 0
				and i in self.available
				and i not in self.blocked
				and (late or self._trusted(i, endgame))
			]
			v0 = dp.pool_value(options, rounds_left - 1)
		k = dp.chances(rnd, rounds_left, self.unmet) / DP_ROUNDS_PER_CHANCE
		if DP_K_CAP:
			k = min(k, DP_K_CAP)
		return dp.values(k, v0)

	def _dp_choose(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		dp = self.dp
		dp.note_singles(turn.round, turn.singles)
		listed_round, listed = self.listed
		if listed_round == turn.round - 1 and listed_round not in self.dated_in:
			for i in listed:
				if i in self.scores:
					dp.note_noshow(i)
		v, v_prev = self._dp_value(turn.round, turn.rounds_left, None)
		fresh_value = v - v_prev
		endgame = self._endgame(turn.rounds_left)
		relaxed = self._relaxed(turn.rounds_left, dp.score_bar(v))

		worth: list[tuple[float, int, float]] = []
		for c in candidates:
			i = c.index
			if i not in self.scores or i in self.blocked:
				continue
			score = self.scores[i][0]
			if score <= 0 or self._dp_paused(i, turn.round):
				continue
			if not self._eligible(i, relaxed, endgame):
				continue
			u = dp.utility(score)
			worth.append((self._p_commit(i) * (u - v_prev), i, u))
		worth.sort(key=lambda w: (-w[0], w[1]))
		if DP_LIST == 'bar':
			good = [i for _, i, u in worth if u >= v]
		else:
			good = [i for value, i, _ in worth if value > 0 and value >= fresh_value]
		fresh = sorted(
			(c for c in candidates if c.index not in self.reports),
			key=lambda c: (self._ad_cost(c.ad), c.index),
		)
		chosen = good + [c.index for c in fresh]
		if not fresh:
			taken = set(good)
			chosen += [i for _, i, _ in worth if i not in taken]
		chosen = chosen[: self.list_limit]
		self.listed = (turn.round, set(chosen))
		return chosen

	def _dp_decide(self, date: DateContext, score: float, first_date: bool) -> Decision:
		dp = self.dp
		partner = date.partner
		self.dated_in.add(date.round)
		self.views[partner] = self._their_view(partner)
		if first_date:
			mutual = not DP_MUTUAL_X or self.views[partner] > 0
			dp.add_sample(score if mutual else NO_CHANCE, date.round)
			if len(self.samples) < DP_PRIOR_N:
				dp.set_prior(self._synthetic_scores(), DP_PRIOR_N)
		# A non-positive match is never better than staying single.
		if score <= 0:
			self.blocked.add(partner)
			return Decision(incompatible=True)

		rounds_left = date.rounds_left
		v, v_prev = self._dp_value(date.round, rounds_left, partner)
		endgame = self._endgame(rounds_left)
		relaxed = self._relaxed(rounds_left, dp.score_bar(v))
		eligible = self._eligible(partner, relaxed, endgame)
		willing = date.round - self.proposed_to_me.get(partner, -100) <= WILLING_ROUNDS
		u = dp.utility(score)
		slack = DP_DELTA_SHARE * self.bonus if willing else 0.0
		accept = eligible and u >= v - slack and score > self._crowd_floor()
		again = accept
		if accept and DP_AGAIN == 'worth' and not willing:
			again = self._p_commit(partner) * (u - v_prev) >= v - v_prev
		if self._dp_paused(partner, date.round):
			again = False
		propose = accept and date.date_number >= 2

		if first_date and accept:
			self.wanted.add(partner)
		elif partner in self.wanted:
			self.returned.add(partner)
		if propose:
			self.proposed_to.add(partner)
		dp.note_decision(partner, again, propose)
		return Decision(again=again, propose=propose)

	# --------------------------------------------------------------- scoring

	def _reprice(self, partner: int) -> tuple[float, float]:
		"""(score after pricing suspicious claims pessimistically, how much that cost)."""
		reports = self.reports[partner]
		ad = self.ads[partner]
		flattered = all(tuple(ideal) == tuple(self.profile) for _, ideal in reports)

		mine = 0.0
		for dim in range(self.dims):
			options = {obs[dim] for obs, _ in reports}
			if ad.own_dim == dim:
				options.add(ad.own_value)  # ad and behaviour disagree: take the worse
			cost = max(self._gap(dim, self.ideal[dim], value) for value in options)
			if flattered and dim == self.my_ad.seek_dim:
				# Average penalties, not traits, so population variance contributes.
				cost = max(cost, self._typical_cost(dim))
			mine += cost

		honest_ideals = [ideal for _, ideal in reports if tuple(ideal) != tuple(self.profile)]
		if honest_ideals:
			theirs = max(decrement(ideal, self.profile, self.monotonic) for ideal in honest_ideals)
		else:
			theirs = self._typical_their_cost()
		score = self.bonus - mine - theirs / 2

		shown = max(
			self.bonus
			- decrement(self.ideal, obs, self.monotonic)
			- decrement(ideal, self.profile, self.monotonic) / 2
			for obs, ideal in reports
		)
		return score, max(0.0, shown - score)

	def _worst_case(self, partner: int) -> float:
		"""Score if the partner used a consistent lie we cannot see on a date.

		Two such lies are common in this class: nudging the trait our ad asks for
		up to 2 steps toward what we want, and showing extreme values as milder
		ones (0-1 as 2, 9-10 as 8, a low monotonic value as 3). Each shown value
		is replaced by the worst true value those lies could hide.
		"""
		score, _ = self.scores[partner]
		reports = self.reports[partner]
		extra = 0.0
		for dim in range(self.dims):
			shown = {obs[dim] for obs, _ in reports}
			worst_shown = max(self._gap(dim, self.ideal[dim], v) for v in shown)
			hidden = set(shown)
			for v in shown:
				if dim == self.my_ad.seek_dim:
					hidden.update(range(max(0, v - 2), min(10, v + 2) + 1))
				if dim < self.monotonic and v == 3:
					hidden.update((0, 1, 2))
				if dim >= self.monotonic and v == 2:
					hidden.update((0, 1))
				if dim >= self.monotonic and v == 8:
					hidden.update((9, 10))
			worst = max(self._gap(dim, self.ideal[dim], v) for v in hidden)
			extra += worst - worst_shown
		return score - extra

	def _robust_unsafe(self, partner: int, endgame: bool) -> bool:
		"""A suspect whose worst case under the lies we suspect is not good enough."""
		if not self.robust:
			return False
		nudged = self.scores[partner][0] - self.robust.nudge_worst(partner)
		gate = ROBUST_NUDGE_GATE or (CROWD_NUDGE_GATE and self._crowd_on(needy=True))
		if gate and not endgame and nudged <= 0:
			return True
		if ROBUST_GATE == 'off' or (ROBUST_GATE == 'endgame' and not endgame):
			return False
		if partner not in self.worst:
			return False
		return self.worst[partner] <= ROBUST_GATE_MARGIN * self.bonus

	def _learn(self, partner: int, observed: Profile, stated_ideal: Profile) -> None:
		"""Record what a typical person looks like, skipping reports with red flags."""
		ad = self.ads[partner]
		if tuple(stated_ideal) == tuple(self.profile):
			return
		if observed[ad.own_dim] != ad.own_value:
			return
		for dim, value in enumerate(observed):
			self.dim_values[dim].append(value)
			self.ideal_values[dim].append(stated_ideal[dim])
		self.their_costs.append(decrement(stated_ideal, self.profile, self.monotonic))

	def _typical_cost(self, dim: int) -> float:
		"""Expected mismatch against observed traits, with a uniform fallback.

		The hidden P is unavailable. With no usable observations, every legal
		value receives equal weight; otherwise use the empirical distribution.
		"""
		values = self.dim_values[dim] or VALUES
		return sum(self._gap(dim, self.ideal[dim], value) for value in values) / len(values)

	def _typical_their_cost(self) -> float:
		if self.their_costs:
			return sum(self.their_costs) / len(self.their_costs)
		# Unknown desired traits go in the "want" argument. This direction
		# matters for monotonic traits, where only unmet desires incur a cost.
		cost = 0.0
		for dim, have in enumerate(self.profile):
			values = self.dim_values[dim] or VALUES
			cost += sum(self._gap(dim, want, have) for want in values) / len(values)
		return cost

	# ------------------------------------------------------------ thresholds

	def _endgame(self, rounds_left: int) -> bool:
		return self.urgent or rounds_left <= max(2, ENDGAME_FRACTION * self.rounds)

	def _floor(self, rounds_left: int) -> float:
		return 0.0 if self._endgame(rounds_left) else self.floor

	def _bar(self, rounds_left: int) -> float:
		"""A personal reservation score that falls as rounds run out or the market empties."""
		if len(self.samples) >= MIN_SAMPLES:
			ordered = sorted(self.samples)
			if self.bar_mode == 'reservation':
				# Optimal stopping: with k chances left, demand the score that at
				# least one of them beats with even odds.
				chances = self._commit_rate() * min(self.unmet, rounds_left // 2)
				if chances < 1:
					return self._floor(rounds_left)
				tail = 1 - 0.5 ** (1 / chances)
				bar = ordered[int((1 - tail) * (len(ordered) - 1))]
				return max(self._floor(rounds_left), bar)
			base = ordered[int(BAR_PERCENTILE * (len(ordered) - 1))]
		else:
			base = BAR_START_SHARE * self.bonus
		left = max(0.0, (rounds_left - 1) / max(1, self.rounds - 1))
		if MARKET_CLOCK:
			left = min(left, self.singles_left)
		return max(self._floor(rounds_left), base * left)

	def _commit_rate(self) -> float:
		if not ADAPTIVE_COMMIT:
			return COMMIT_RATE
		prior = PRIOR_WEIGHT * COMMIT_RATE
		return (len(self.returned) + prior) / (len(self.wanted) + PRIOR_WEIGHT)

	def _tolerance(self, endgame: bool) -> float:
		share = LIE_TOLERANCE_ENDGAME_SHARE if endgame else LIE_TOLERANCE_SHARE
		return share * self.bonus

	def _trusted(self, partner: int, endgame: bool) -> bool:
		return self.scores[partner][1] <= self._tolerance(endgame)

	def _eligible(self, partner: int, relaxed: bool, endgame: bool) -> bool:
		return relaxed or self._trusted(partner, endgame)

	def _relaxed(self, rounds_left: int, bar: float) -> bool:
		if rounds_left <= (1 - RELAX_FRACTION) * self.rounds:
			return True
		trusted = sum(
			1
			for i, (score, _) in self.scores.items()
			if i in self.available
			and i not in self.blocked
			and score >= bar
			and self._trusted(i, False)
		)
		return trusted < MIN_TRUSTED_OPTIONS

	def _estimate_options(self, candidates: tuple[Candidate, ...]) -> float:
		pool = list(candidates)
		if len(pool) > 200:
			pool = self.rng.sample(pool, 200)
		hits = 0
		for c in pool:
			ad = c.ad
			for _ in range(SCARCE_DRAWS):
				mine = theirs = 0
				for dim in range(self.dims):
					have = ad.own_value if dim == ad.own_dim else self.rng.choice(VALUES)
					want = ad.seek_value if dim == ad.seek_dim else self.rng.choice(VALUES)
					mine += self._gap(dim, self.ideal[dim], have)
					theirs += self._gap(dim, want, self.profile[dim])
				hits += self.bonus - mine - theirs / 2 > 0
		return hits / SCARCE_DRAWS * len(candidates) / max(1, len(pool))

	def _early_scarce(self) -> bool:
		return bool(SCARCE_OPTIONS) and self.options is not None and self.options < SCARCE_OPTIONS

	def _scarce(self) -> bool:
		"""Few of our first dates would make us happy: take any positive match."""
		if not self.scarce_share or len(self.samples) < MIN_SAMPLES:
			return False
		positive = sum(1 for s in self.samples if s > 0) / len(self.samples)
		return positive < self.scarce_share

	def _unresponsive(self, partner: int) -> bool:
		return self.ignored.get(partner, 0) >= MAX_IGNORED

	def _best_other(self, partner: int, relaxed: bool, endgame: bool) -> float:
		"""Best score among other people we've dated who still look available."""
		scores = [
			score
			for i, (score, _) in self.scores.items()
			if i != partner
			and i in self.available
			and i not in self.blocked
			and not self._unresponsive(i)
			and self._eligible(i, relaxed, endgame)
			and (not self.engaged_only or self.met.get(i, 0) >= 2 or i in self.proposed_to_me)
		]
		return max(scores, default=float('-inf'))

	# ------------------------------------------------------------- utilities

	def _make_ad(self) -> Advertisement:
		return self._ad_for(self.profile, self.ideal)

	def _ad_for(self, profile: Profile, ideal: Profile) -> Advertisement:
		"""The ad our rule posts for someone with this profile and ideal.

		With COORDINATE, the seek half is replaced by a signature our copies
		recognise; a copy's date reproduces its ad exactly through this method.
		"""

		# Honest ad: our best monotonic dimension, and the dimension where our
		# ideal is most extreme (the one a random partner is likeliest to miss).
		def centred(value: int) -> int:
			return 10 - 2 * abs(value - 5)

		if AD_OWN == 'appealing':
			# High is good on monotonic traits; near the middle suits most people otherwise.
			own = max(
				range(self.dims),
				key=lambda i: profile[i] if i < self.monotonic else centred(profile[i]),
			)
		else:
			showable = range(self.monotonic) if self.monotonic else range(self.dims)
			own = max(showable, key=lambda i: profile[i])
		if AD_SEEK == 'lenient':
			seek = max(
				range(self.dims),
				key=lambda i: 10 - ideal[i] if i < self.monotonic else centred(ideal[i]),
			)
		else:
			seek = max(range(self.dims), key=lambda i: abs(ideal[i] - 5))
		seek_value = ideal[seek]
		if self.coordinating:
			seek, seek_value = signature(own, profile[own])
		return Advertisement(
			own_dim=own, own_value=profile[own], seek_dim=seek, seek_value=seek_value
		)

	def _ad_cost(self, ad: Advertisement) -> float:
		"""What the two advertised dimensions alone would cost us."""
		mine = self._gap(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		theirs = self._gap(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		return mine + theirs / 2

	def _gap(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		if dim < self.monotonic and gap <= 0:
			return 0
		return gap * gap
