
import os
import joblib
import numpy as np
import pandas as pd
import warnings


class BiddingAgent:

    def __init__(self):

        # ── Money management ──
        self.bankroll = 500000.0
        self.initial_bankroll = 500000.0
        self.total_wins = 0
        self.estimated_profit = 0.0
        self.predicted_value = 0.0
        self.round_number = 0
        self.total_spent = 0.0

        # Load all pkl files using absolute path
        base_path = os.path.dirname(os.path.abspath(__file__))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = joblib.load(
                os.path.join(base_path, 'model_Medhavi.pkl'))
            self.encoders = joblib.load(
                os.path.join(base_path, 'encoders_Medhavi.pkl'))
            self.imputation_dicts = joblib.load(
                os.path.join(base_path, 'imputation_dicts.pkl'))
            self.feature_constants = joblib.load(
                os.path.join(base_path, 'feature_constants.pkl'))

    def analyze_item(self, item_features: dict):
        self.round_number = 0
        features = self._preprocess(item_features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.predicted_value = float(
                self.model.predict(features)[0])

    def place_bid(self, current_highest_bid: float) -> float:
        self.round_number += 1

        if self.predicted_value <= 0:
            return 0.0

        # ════════════════════════════════════════════
        # DETERMINISTIC BID SEQUENCE
        # Next bid is a pure function of:
        #   self.predicted_value
        #   self.bankroll
        #   current_highest_bid
        #   self.round_number (n)
        # ════════════════════════════════════════════

        # ── LAYER 1: Dynamic ceiling ──
        # Base margin depends on car value tier
        # More expensive car = more profit room = bid higher
        if self.predicted_value >= 20000:
            base_margin = 0.08    # ceiling = 92% of value
        elif self.predicted_value >= 10000:
            base_margin = 0.10    # ceiling = 90% of value
        else:
            base_margin = 0.12    # ceiling = 88% of value

        # Bankroll ratio — pure function of self.bankroll
        bankroll_ratio = self.bankroll / self.initial_bankroll

        # Bankroll penalty — conservative when poor
        if bankroll_ratio < 0.20:
            bankroll_penalty = 0.07
        elif bankroll_ratio < 0.40:
            bankroll_penalty = 0.03
        else:
            bankroll_penalty = 0.0

        # Profit bonus — aggressive when winning
        profit_bonus = min(
            self.estimated_profit / 1_000_000, 0.04)

        # Final margin — clamped 6% to 22%
        margin = base_margin + bankroll_penalty - profit_bonus
        margin = max(0.06, min(0.22, margin))

        # Hard ceiling
        max_bid = self.predicted_value * (1.0 - margin)

        # ── LAYER 2: Walk away conditions ──

        # Condition 1: rival already above our ceiling
        if current_highest_bid >= max_bid:
            return 0.0

        # Condition 2: per-car budget cap (pure function of bankroll)
        if bankroll_ratio >= 0.60:
            per_car_cap = self.bankroll * 0.20
        elif bankroll_ratio >= 0.30:
            per_car_cap = self.bankroll * 0.12
        else:
            per_car_cap = self.bankroll * 0.07

        if max_bid > per_car_cap:
            return 0.0

        # Condition 3: bankroll exhausted
        if current_highest_bid >= self.bankroll:
            return 0.0

        # Condition 4: session spending cap
        # Never spend more than 65% of initial bankroll total
        # When we've spent a lot, only chase exceptional cars
        total_spent = self.initial_bankroll - self.bankroll
        if total_spent > self.initial_bankroll * 0.65:
            if self.predicted_value < 25000:
                return 0.0   # only bid on high-value cars now

        # ── LAYER 3: Deterministic increment formula ──
        # gap = distance between current bid and our ceiling
        gap = max_bid - current_highest_bid

        # Aggression constant X — pure function of self.bankroll
        if bankroll_ratio >= 0.60:
            X = 0.45
            min_increment = 400.0
        elif bankroll_ratio >= 0.30:
            X = 0.30
            min_increment = 200.0
        else:
            X = 0.15
            min_increment = 100.0

        # Core formula: increment(n) = gap × (X / n)
        # Geometric decay — aggressive early, gentle late
        increment = max(
            gap * (X / self.round_number),
            min_increment
        )

        # ── LAYER 4: Snipe override ──
        # If round ≥ 3 AND gap still > 20% of ceiling
        # → rival is weak → close deal instantly
        if self.round_number >= 3 and gap > (max_bid * 0.20):
            increment = gap * 0.75

        # Calculate next bid
        next_bid = current_highest_bid + increment

        # Hard ceiling — NEVER cross (mathematical guarantee)
        next_bid = min(next_bid, max_bid)

        return round(next_bid, 2)

    def auction_result(self, won: bool, winning_bid: float,
                       actual_price: float,
                       current_bankroll: float):
        # Arena gives official bankroll — always use it
        self.bankroll = current_bankroll

        if won:
            self.total_wins += 1
            self.total_spent += winning_bid
            profit_estimate = self.predicted_value - winning_bid
            self.estimated_profit += profit_estimate
            print(
                f"Won! Paid ${winning_bid:,.0f} | "
                f"Predicted ${self.predicted_value:,.0f} | "
                f"Est. profit ${profit_estimate:,.0f} | "
                f"Bankroll ${self.bankroll:,.0f}"
            )

    def _preprocess(self, car: dict) -> pd.DataFrame:
        c = car.copy()

        # STEP 1: Clean text
        text_cols = ['make', 'model', 'trim', 'body',
                     'transmission', 'color', 'interior', 'state']
        for col in text_cols:
            val = c.get(col, None)
            if val is None or (
                    isinstance(val, float) and np.isnan(val)):
                c[col] = 'unknown'
            else:
                c[col] = str(val).lower().strip()

        # STEP 2: Fix body variants
        body_map = {
            'crew cab': 'pickup', 'extended cab': 'pickup',
            'supercab': 'pickup', 'regular cab': 'pickup',
            'cab plus': 'pickup', 'double cab': 'pickup',
            'king cab': 'pickup', 'quad cab': 'pickup',
        }
        c['body'] = body_map.get(c['body'], c['body'])
        if c['body'] in ['unknown', '']:
            c['body'] = self.imputation_dicts['body_mode']

        # STEP 3: Fix transmission
        if c['transmission'] in ['unknown', 'nan', '']:
            c['transmission'] = 'automatic'

        # STEP 4: Fix odometer
        odo = c.get('odometer', None)
        if odo == 999999 or odo is None or (
                isinstance(odo, float) and np.isnan(odo)):
            odo = None
        if odo is None:
            key = (c['make'], int(c.get('year', 2010)))
            c['odometer'] = float(
                self.imputation_dicts['odometer_medians'].get(
                    key,
                    self.imputation_dicts['odometer_global_median']
                ))
        else:
            c['odometer'] = float(odo)
        c['odometer'] = min(c['odometer'], 400000.0)

        # STEP 5: Fix condition
        cond = c.get('condition', None)
        if cond is None or (
                isinstance(cond, float) and np.isnan(cond)):
            key = (c['make'], int(c.get('year', 2010)))
            c['condition'] = float(
                self.imputation_dicts['condition_medians'].get(
                    key,
                    self.imputation_dicts['condition_global_median']
                ))
        else:
            c['condition'] = float(cond)

        # STEP 6: Fill unknowns
        for col in ['make', 'model', 'trim', 'color', 'interior']:
            if c[col] in ['', 'nan']:
                c[col] = 'unknown'

        # STEP 7: Engineer features
        AUCTION_YEAR = self.feature_constants['AUCTION_YEAR']
        LUXURY_BRANDS = self.feature_constants['LUXURY_BRANDS']

        c['car_age'] = max(
            AUCTION_YEAR - int(c.get('year', 2010)), 1)
        c['usage_intensity'] = (
            c['odometer'] / (c['car_age'] + 1))
        c['age_odometer_interaction'] = (
            c['car_age'] * c['odometer'])
        c['is_luxury'] = (
            1 if c['make'] in LUXURY_BRANDS else 0)
        c['trans_body'] = (
            c['transmission'] + '_' + c['body'])
        c['condition_age_ratio'] = (
            c['condition'] / (c['car_age'] + 1))

        # STEP 8: Encode text → numbers
        for col in ['make', 'model', 'trim', 'body',
                    'transmission', 'state', 'color',
                    'interior', 'trans_body']:
            if col in self.encoders:
                val = c[col]
                encoder = self.encoders[col]
                if val in encoder.classes_:
                    c[col] = int(encoder.transform([val])[0])
                else:
                    c[col] = 0   # unseen value fallback

        # STEP 9: Return DataFrame in exact training order
        feature_order = [
            'make', 'model', 'trim', 'body', 'transmission',
            'state', 'condition', 'odometer', 'color', 'interior',
            'car_age', 'usage_intensity', 'age_odometer_interaction',
            'is_luxury', 'trans_body', 'condition_age_ratio'
        ]
        return pd.DataFrame(
            [[c[feat] for feat in feature_order]],
            columns=feature_order
        )
