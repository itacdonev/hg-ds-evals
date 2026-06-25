# Report 2 — test-set expansion: new questions & KB gaps
**Inputs:** the 198-fragment KB and the existing 604-row test set (`SLSP_test_cases.csv`).
**Deliverable:** `new_test_cases.csv` — proposed questions in the **exact SLSP schema** (SK + EN), directly appendable; gap-exposing cases are left unanswered with the missing-information area noted in `comment`. This report explains and prioritises them.
> Per your instruction I did **not** draft `expected_answer_SK` for the answerable additions — those need an authoritative source. Gap cases have no expected answer **by design**: the point is that the KB, as written, doesn't fully answer them.

> **Important framing (corrected).** Exact fees/rates/limits are *intentionally* not in the KB — they live in the official price list (Sadzobník), shown to the customer as a **clickable link they open themselves**. So fee/rate/limit questions are **not knowledge gaps**; they are *link-expected* test cases (the assistant should surface the right link). They are tagged `[LINK-EXPECTED]` in the CSV and counted separately below. The genuine knowledge gaps are the rest (missing procedures, eligibility, conditions), tagged `[KB-GAP]`. Where an auto-generated line below says the agent *cannot read* a PDF/link or that a rate is *unanswerable by design*, read it as: *the figure isn't restated in the ENUM fragment — it lives in the linked price list the customer opens.*

## Method
For every fragment, the content pass proposed up to 3 *additional* high-probability user questions that are **not** already in the test set (deduplicated against the existing questions per topic), prioritising ones the KB **cannot** fully answer. Each is tagged `answerable_by_kb` = `yes` / `partial` / `no`, a `priority`, and — when not fully answerable — a `missing_area`.

## Summary
- **398** proposed questions across **198** fragments.
- **209 are genuine knowledge gaps** (`[KB-GAP]`) — missing procedures, eligibility rules or conditions the KB should hold but doesn't. **These are the ones to fill.**
- **125 are link-expected** (`[LINK-EXPECTED]`) — fees/rates/limits whose answer is the price-list link by design; use them to test that the assistant surfaces the right link, not to write KB content.
- **64** are fully answerable today — useful as extra coverage/regression tests.
- By priority: HIGH 125, MED 237, LOW 36.

## How to use `new_test_cases.csv`
- Columns match `SLSP_test_cases.csv` exactly; `test_case_number` uses a `PROPOSED-NNNN` prefix so the rows are distinguishable from validated cases until you accept them.
- `comment` encodes `PROPOSED <priority>/<answerable_by_kb>`, a tag (`[KB-GAP]` / `[LINK-EXPECTED]` / `[answerable]`), the `missing_area`, and a one-line rationale.
- **Triage suggestion:** `[KB-GAP]` rows drive KB content edits; `[LINK-EXPECTED]` rows test that the assistant returns the right price-list link (and flag any ENUM fragment missing that link); `[answerable]` rows can be merged into the eval set as-is after a gold-answer pass.
- Rows are sorted so gap-exposing + HIGH-priority appear first.

## 1. Missing-information areas, by product family
Each family below lists the **distinct information areas** the KB is missing (named so colleagues can fill them), with the count of gap-exposing questions that hit that family.

### GIRO — 43 gap questions
- Whether and how an account can be opened for a child under 6 years
- The actual fee for a second/additional card (only a link to the fee schedule is given)
- Monthly account maintenance fee / conditions for it to be free
- Prerequisites for account closure (balance, linked cards/products, overdraft)
- The specific scope of operations permitted to a disponent
- List of available foreign-currency account currencies
- account-opening fee for the foreign-currency account
- incoming/outgoing foreign-currency transfer and conversion fees
- monthly card fee amount
- list of currencies in which the foreign-currency account can be opened
- minimum age to open a foreign-currency account
- definition of a qualifying 'regular payment' (standing order / direct debit / minimum amount)
- …and 31 more (see CSV).

### LOAN — 42 gap questions
- Late/missed mortgage payment consequences (late interest, penalties, recovery)
- Approval/processing time (lives only in MORTGAGE_ABOUT, not in this application fragment)
- First-drawdown deadline (the 6/36-month figure sits in MORTGAGE_POST_SIGNING_CONDITIONS, not here)
- Actual current interest-rate figure (KB only links to an external PDF the customer opens it via a link)
- Income documentation for applicants working abroad (this fragment covers employees, pensioners and self-employed, but not foreign-income applicants — unlike LOAN@UNSECURED_APPLI...
- Maximum number of loans/obligations that can be consolidated
- Concrete/indicative starting interest rate for the refinancing loan
- Fees/refunds tied to cancelling a loan application (especially after signing the draft contract)
- Changing the mortgage due/debit date
- A concrete phone number / direct line (fragment only offers the in-app contact form and 'Mám záujem' callback)
- Whether property insurance (and its vinkulácia) is a pre-drawdown condition
- Document checklist specific to refinancing (e.g. payoff statement from old bank, existing loan contract)
- …and 30 more (see CSV).

### CARD_CREDIT — 24 gap questions
- deactivation/modification of the automatic full-repayment service
- credit-card interest rate / APR
- concrete eligibility conditions (age range, income/creditworthiness)
- decision/approval timeframe
- ATM cash-withdrawal fee for credit cards
- maximum number of credit cards per client
- proactive status-check channel (George / Klientske centrum) and decision timeframe
- replacement-card channel when George app is unavailable
- fee for duplicate/extraordinary statement
- whether/how to re-enable postal statements after George auto-switches to e-statements
- consequence of not repaying in full before the notice period ends
- processing time for a credit-line increase
- …and 12 more (see CSV).

### SAVING — 23 gap questions
- Exact EUR fee for cash deposit/withdrawal at the branch for a savings account
- Actual current rate per tenor/currency (only an external PDF and a hypothetical 1,5 % example exist)
- Interest consequence of releasing locked funds mid-term
- Treatment of round-ups on refunded/returned card transactions
- Fee (or fee-free statement) for setting up/cancelling a disponent
- Processing/settlement time for a transfer from savings to own/other account
- Minimum age for self-service opening via George (held only in SAVING@ABOUT, not here)
- Age eligibility (6-14) is absent from the application fragment
- Maximum balance / limit on Pokladnička
- Whether the 1% Pokladnička rate is fixed/guaranteed or variable
- Daily/monthly cap on rounded-up amounts
- Activation time after online opening
- …and 11 more (see CSV).

### CARD_DEBIT — 21 gap questions
- Blocking card transactions of a specific merchant (promised in summary, absent from description)
- Consistent ePIN length (fragment says both four and five digits)
- Foreign-transaction fees and currency-conversion terms
- image submission/upload channel and approval notification
- foreign-currency / cross-border transaction fee amount
- monthly debit card fee amount
- Issuance fee for a new debit card
- Re-delivery time for a replacement card
- monthly/annual debit card fee (this fragment lists benefits only, no price; would have to route to CARD_DEBIT@FEES)
- behaviour on repeated wrong ePIN entry (lockout / number of attempts / how to recover besides setting a new one)
- price of a custom-image / redesigned card
- out-of-network ATM withdrawal fee
- …and 9 more (see CSV).

### INSURANCE — 16 gap questions
- Description and differentiation of the two named life products
- Price and sum-insured / payout amounts for life insurance
- Price/premium of Poistenie Domov
- How to cancel/terminate the policy or stop the yearly auto-renewal
- Deductible (spoluúčasť) amount for OVAK
- Whether non-death risks (disability, critical illness) are covered
- Treatment of a child's cover when they reach the 18-year age limit
- Clear distinction between the 200 € cap for ordinary cash theft and full reimbursement only under violent robbery
- A single unambiguous price for the risky-sports add-on (fragment gives both 15 €/30 € via HZS and 36 €/60 € standalone)
- Concrete claim/reporting channel and any deadline for the undelivered-online-goods risk (only 'contact the seller and cooperate' is given)
- Main-policy deductible amount (only the 10% bicycle deductible is given)
- Whether household-contents and building cover can be bought separately or only together
- …and 4 more (see CSV).

### ATM — 13 gap questions
- concrete daily / per-withdrawal ATM cash-withdrawal limit in EUR
- actual fee amounts for other-bank ATMs in SK and abroad
- AML threshold amount/period for cash deposits without proof of origin
- card-payment (POS/e-commerce) dispute/chargeback process — the fragment covers only ATM withdrawals despite its name
- fate of payment/standing orders that were previously created via ATM after the service was discontinued on 1 Jan 2026
- number of incorrect PIN attempts that triggers a block
- outcome for a retained card (returned vs destroyed vs reissue required) and timeline
- branch ('pobočka') as an alternative channel for transfers (only George is mentioned)
- foreign-currency cash withdrawal/ordering at a branch (summary promises it, description does not cover it)
- which exact account types have fee-free deposits (text says both 'osobné' and 'SPACE')
- limits/fees when depositing to an account other than the cardholder's own
- behaviour of an expired/unused mobile-withdrawal code and its effect on daily limits
- …and 1 more (see CSV).

### GEORGE — 7 gap questions
- Whether chat/conversation history is stored
- Exact per-SMS notification price (only a tariff PDF link is given)
- Mobile-app notification setup (this fragment is explicitly web-only)
- Whether hiding a product affects its functionality/notifications (KB only covers visibility toggling)
- A working privacy/security reference link
- Complaints/reklamácie are explicitly out of scope here; needs routing to the complaints fragment
- How access-only (non-owned) products appear and need to be enabled in 'Prispôsobiť prehľad'

### START_TRANSFER — 6 gap questions
- cancellation of an already-sent cross-border payment (summary promises 'ako zrušiť cezhraničnú platbu' but body only covers non-delivery investigation)
- overall daily/per-payment transfer limit (only the instant-payment limit lives in INSTANT_TRANSFER; standard transfer limit and branch EUR limit are absent here)
- concrete fee/range for a generic cross-border (SWIFT) payment
- explicit yes/no on anonymous payments
- wrongly-entered payment handling (explicitly out of scope here; lives in TRANSACTION_CLAIM)
- current (post-sanctions) availability of RUB cross-border payments

### PORTFOLIO — 6 gap questions
- Procedure for closing the majetkový účet (summary promises it; description omits it).
- Contradiction between PORTFOLIO@ABOUT (branch-only redemption) and SELL_ORDER (in-app 'Predať').
- Per-trade buy/sell commission.
- Required documents/identification for opening.
- Withdrawing/transferring cash out of the majetkový účet.
- Whether the 1-year time-test exemption applies to mutual-fund units or only to exchange-traded securities ('cenné papiere obchodované na burze').

### DEBT — 4 gap questions
- Frequency limit for changing the repayment date
- Whether the next-business-day grace applies to credit cards (and overdraft) or only to loans
- Application channel/steps and eligibility criteria for the temporary 25% reduction
- The current životné minimum figure; only the §104 one-off amounts (99.58€ / 165€) are quantified, and those may be outdated.

### DATEIO — 4 gap questions
- Claim filing deadline / time window
- Effect of returning/refunding a purchase on the Moneyback reward (present in CLAIMS, absent here)
- Any minimum-spend or maximum-reward limits (the fragment only says amounts vary per offer/merchant)
- Concrete resolution SLA (only 'a few days' is stated)

### ACCOUNT_STATEMENT_LIST — 4 gap questions
- Exact EUR fee for postal/paper statements (fragment only links the sadzobnik PDF, gives no number)
- Whether recurring English statements are possible at all (body implies only one-off requests); summary contradicts body
- Exact archive retention limit; '18 mesiacov' vs unspecified 'obmedzene zobrazovanie'
- Concrete availability timing (only 'po skonceni kalendarneho roka')

### CHANGE_CARD_LIMITS — 4 gap questions
- maximum daily ATM withdrawal limit in EUR
- Maximum allowed online/card limit value
- maximum internet-payment limit value and its cap relative to the merchant limit
- Non-limit/non-balance decline causes (3D Secure, fraud hold, card status, merchant side)

### ACCOUNT — 3 gap questions
- The fragment contradicts itself on whether a branch visit is mandatory, so it cannot give a single deterministic answer.
- Causes of a negative balance other than an authorised overdraft (fees, unauthorised overdraft, late card settlement).
- Default disponent limits / how the limit amount is determined.

### TRANSACTION_CLAIM — 3 gap questions
- concrete fee for a payment-return request
- concrete expected arrival time for standard incoming payments
- obligation to return mistakenly-received funds (summary promises this, body only shows how to return, not whether you must)

### OUTGOING_TRX_CLAIM — 3 gap questions
- claim filing deadlines for card payments
- card-claim resolution timeframe
- how to block/stop recurring card charges (promised in summary, not in body)

### PENSION — 3 gap questions
- Tax-deduction treatment for pre-1.1.2014 contracts.
- Amount/cap and frequency of the early withdrawal; KB only gives the 10-year eligibility.
- Setup/management fees; only the first-year transfer fee (max 5 %) is documented.

### FRAUD — 3 gap questions
- The process and conditions for recovering / refunding fraudulently taken funds.
- Steps to block the account / online banking / George access (the description only covers blocking the card).
- Guidance distinguishing a small unrecognised charge (e.g. card verification / subscription) from actual fraud; the description treats any unknown payment uniformly.

### ACTIVATION_CODE — 2 gap questions
- activation code validity period / expiry time
- recovery path for an invalid/expired code beyond 'start again with a new code' (e.g. how many attempts, what triggers invalidity)

### SHARE_PRODUCTINFO — 2 gap questions
- Whether the loan-account IBAN (listed under loan Info) can itself be shared the same way as the current-account IBAN, or only via the generated confirmation
- The complete field list contained in the shared account-info package

### PRODUCTDETAIL_CASHFLOW_GRAPH — 2 gap questions
- Whether recategorising one payment changes the rule for future payments from the same merchant
- Which transaction types/accounts feed the graph (covered in PFM_SETTINGS, not here)

### PFM_BUDGETS — 2 gap questions
- Whether budgets trigger any alert/notification on overspend
- Troubleshooting when budgets do not appear (e.g. account not included in PFM, no spend yet)

### PFM_SETTINGS — 2 gap questions
- Effect of removing an account on already-categorised historical data
- Treatment of debit cards (only credit card is explicitly listed)

### PFM_SPENDING_BUDGET — 2 gap questions
- Whether a single total spending budget (separate from category budgets) exists and how to set it
- Maximum number of simultaneous budgets

### PFM_SUBCATEGORY_BUDGET — 2 gap questions
- Relationship/aggregation between subcategory budgets and the parent category budget
- The initial create/set flow for a subcategory budget (only edit/cancel is described)

### APP_NOTIFICATIONS — 2 gap questions
- How to enable/disable individual notification types (vs whole accounts) on mobile
- Deeper troubleshooting (battery optimisation, Do Not Disturb, re-login, contact support)

### BOOK_APPOINTMENT — 2 gap questions
- Booking branch appointments for non-investment topics
- Cancelling/rescheduling an existing appointment

### GEORGE_DEVICES — 2 gap questions
- Removing/deactivating a device from the list
- Clear activation flow when the old device is unavailable (SMS request / Klientske centrum)

### TOKENIZE_CARD — 2 gap questions
- Troubleshooting failed contactless/mobile payments after tokenization
- Minimum age for Garmin/Xiaomi/Swatch wallets

### SHOW_ATM_BRANCH_FINDER — 2 gap questions
- An in-app way to locate Erste Group ATMs abroad (the fragment defers entirely to an external web link)
- How to identify and locate cash branches ('hotovostné pobočky') for foreign-currency withdrawal

### CREATE_STANDING_ORDER — 2 gap questions
- list of available standing-order frequencies
- whether a standing order can target a savings account (summary promises this, body says setup is only on a 'bežný/osobný účet')

### SHOW_REQUESTS — 2 gap questions
- Whether branch-tablet-signed documents really appear in Úložisko dokumentov (the section duplicates the George-signed text without confirming)
- Steps to share/forward (not just download) a stored document

### TOKEN_MANAGEMENT — 2 gap questions
- Clear statement of Android face-recognition support (description lists face only for iOS but summary mentions Face Unlock on Android)
- Troubleshooting when biometrics is unavailable/greyed out (only says it must first be enabled in the phone settings, no further steps)

### WRITE_MESSAGE — 2 gap questions
- Consultant response time / availability
- Path for clients with no assigned consultant

### SECURE_PHONE_NUMBER — 2 gap questions
- Activation/processing time for the new SMS-key number
- Procedure when the SMS-key phone/number is lost or stolen (KB only covers planned change and abroad/health exceptions, not loss of access)

### CLOSE_ACCOUNT — 2 gap questions
- Treatment and payout of accrued interest at the moment of closing the savings account.
- Fees / penalty and notice period for closing a savings account.

### ROUND_UP_SAVING_APPLICATION — 2 gap questions
- Whether the target/destination account for round-up savings can be changed in these settings.
- Whether full deactivation of Drobné bokom is done in this same place or elsewhere (notInScope excludes cancellation but doesn't point anywhere).

### SHOW_EXTRA_ITEMS — 2 gap questions
- A clear definition contrasting 'čakajúca' vs 'plánovaná' payment (the summary promises it, the description blurs it).
- Explicit cancel steps for a 'čakajúca platba' specifically (the description details cancelling planned payments and unrealized payments, but the waiting-payment case is not sepa...

### CALL_PHONE_AUTHORISED_V2 — 2 gap questions
- Definition and mechanism of the 'verified/authenticated' call (the central topic implied by the name).
- Troubleshooting / fallback when the in-app call fails to launch.

### EASY_ACCESS — 2 gap questions
- Source of the activation code when no other active device exists (branch / Client Centre).
- The concrete steps to reset the George PIN (the fragment says reset is needed but not how).

### LOCK_CARD_PERM — 2 gap questions
- behaviour of recurring payments / direct debits during a temporary block
- whether refunds arrive on a blocked/reissued plastic card (covered for virtual cards elsewhere, not here)

### PFM_CATEGORY_BUDGETS — 2 gap questions
- Budget period — all PFM fragments imply monthly but never state whether weekly/yearly is possible
- Whether budgets reset monthly or roll over / carry unspent amounts

### INSTANT_TRANSFER — 2 gap questions
- fee for sending an instant payment (vs standard SEPA)
- explicit statement that a completed instant payment is irreversible

### SCAN_AND_PAY — 2 gap questions
- support for scanning a postal money order (poukážka) in the one-off payment flow
- whether scanned QR/IBAN fields (amount, VS) can be edited before confirming

### INVEST — 2 gap questions
- Residency eligibility for Slovak residents (the text only lists 'trvalý pobyt na území ČR', which is wrong/contradictory for a Slovak bank).
- Whether minimum entry amounts belong here at all (notInScope excludes them), and minimums for other products are not given.

### CLOSE_CARD — 2 gap questions
- whether merchant refunds reach the customer after a plastic card is cancelled (documented only for virtual cards)
- whether the underlying account remains open when the card contract ends

### REGULAR — 2 gap questions
- Cancellation cut-off / whether the next scheduled execution still runs.
- Whether 20 € is the general regular-investing minimum for all eligible products or only crypto ETP.

### GEORGE_INVEST — 2 gap questions
- Description/comparison of the three portfolio programs (only their names are listed).
- Only a one-line mention exists; no real explanation of the InveStorky content/format.

### SELL_ORDER — 2 gap questions
- Settlement timing for sale proceeds.
- Limit vs Market order type for selling.

### SECURITIES_ORDERS — 2 gap questions
- Cancelling an open (otvorené) investment order.
- Whether 'ukončené' = executed, rejected, or expired.

### BUY_ORDER — 2 gap questions
- Whether the 150 EUR one-off minimum applies to all instruments or only funds.
- Order-type selection for follow-on share purchases (the funds follow-up flow shown here omits Limit/Market).

### SEARCH_SECURITIES — 2 gap questions
- First-time buyer path; the steps here are framed for users who 'už ... vlastníte', and account creation is only implied.
- Whether the 'Svetové akcie' list equals 'top 10 by market cap' (summary) vs just price/daily change (body).

### FX_CONVERTER — 2 gap questions
- fee/margin for buying foreign currency cash (valuty) at the branch
- whether the 2% debit-card surcharge applies to POS payments abroad

### REQUEST_CARD — 2 gap questions
- replacement card fee for the lost/stolen case
- card-number behaviour on auto-renewal (PIN stays; number stays 'in most cases' — the exceptions are not specified)

### BANK_TRANSFER — 2 gap questions
- whether credit products (credit card, overdraft) and balance transfer are part of the switch
- whether incoming payments to the old account number are redirected/forwarded after switching

### TRANSFER_TO_CONTACT — 2 gap questions
- Payme fees and link validity period
- full list of banks supporting Payme (only ČSOB, Tatra banka, VÚB named)

### REPAY_CREDIT_CARD — 2 gap questions
- interest consequences of paying minimum vs full balance / interest-free period rules
- due date / interest-free period length for credit-card repayment

### SK_DIRECTDEBIT — 2 gap questions
- refund deadline for an unauthorised direct debit (vs the 8-week voluntary window)
- explanation of the B2B scheme and its (non-)refundability

### CREATE_SWEEP_ORDER — 2 gap questions
- concrete setup fee amount for an above-limit standing order
- change/cancel procedure for an above-limit standing order

### ACTIVATE_CARD — 2 gap questions
- what to do when the activation button is missing / card already shows active or not yet delivered
- time until the card is usable after activation (immediate vs delay)

### TEMPLATE_TRANSFER — 2 gap questions
- maximum number of saved payment templates (if any)
- cross-reference that templates can seed a standing order (covered in CREATE_STANDING_ORDER, not here)

### DRAWDOWN — 1 gap questions
- timeframe for disbursing an approved consumer loan to the account

### ASSETS — 1 gap questions
- Troubleshooting when valuation/market price is missing or delayed (data refresh timing).

### SHOW_CARDS_OVERVIEW — 1 gap questions
- The actual default display order of cards

### CALL_PHONE — 1 gap questions
- The actual insurance assistance-line phone number (fragment only routes the user to the insurer's website via the claim form).

### DISPLAY_LIMITS — 1 gap questions
- the answer differs for the general daily limit (no) vs the instant-payment limit (yes); the bare question is ambiguous as written

### SHOW_CARD_PAN_CVC — 1 gap questions
- whether details can be re-displayed repeatedly / any limit on how often

### CALL_BRANCH_AUTHORISED — 1 gap questions
- The actual Klientske centrum phone number(s)

### CONTACT_MANAGEMENT — 1 gap questions
- Alternative channel for changing data when no Slovak OP exists

### PRODUCTDETAIL_TRANSACTION_SEARCH — 1 gap questions
- Path to the actual transaction/movements list (fragment only covers the Analyza graph)

### SHARE_TRANSACTION — 1 gap questions
- Legal/official status of the generated PDF confirmation (whether it is bank-certified / accepted as official proof)

### CHANGE_DELIVERY_ADDRESS — 1 gap questions
- Cut-off / deadline for changing the delivery address of an in-production card

### PROFILE — 1 gap questions
- Unambiguous list of permitted special characters in the alias

### PRODUCT_SETTINGS — 1 gap questions
- Effect of hiding a product (display-only vs functional)

### SHOW_CARD_PIN — 1 gap questions
- the actual postal-PIN fee amount (fragment only links to the Sadzobník)

### UNLOCK_CARD_PERM — 1 gap questions
- Next step for a permanently blocked card (replacement path)

### VIRTUAL_CARDS — 1 gap questions
- whether ATM withdrawal requires the phone/wallet (no physical card exists, so no fallback is described)

---
## 2. Highest-priority gap questions (feature list)
HIGH-priority questions the KB cannot fully answer — the ones most worth fixing *and* testing. Full set (398) in `new_test_cases.csv`.

| Fragment | Question (EN) | Missing |
|---|---|---|
| `ACCOUNT@DISPOSING_PERSON` | Do I have to go to a branch to add an authorized person, or can I do it entirely in Geo... | The fragment contradicts itself on whether a branch visit is mandatory, so it cannot gi... |
| `ACCOUNT@REMAINING_BALANCE` | I'm in the negative but I don't have an authorised overdraft — how is that possible? | Causes of a negative balance other than an authorised overdraft (fees, unauthorised ove... |
| `ATM@WITHDRAWAL` | How much can I withdraw from an ATM with my card per day? What is my daily limit? | concrete daily / per-withdrawal ATM cash-withdrawal limit in EUR |
| `ATM@WITHDRAWAL` | How much will I pay for a withdrawal from another bank's ATM in Slovakia, and how much ... | actual fee amounts for other-bank ATMs in SK and abroad |
| `ATM@DEPOSIT` | How much can I deposit via the deposit ATM without having to document the origin of the... | AML threshold amount/period for cash deposits without proof of origin |
| `ATM@TRANSACTION_CLAIM` | A merchant charged my card twice — how do I file a claim for that? | card-payment (POS/e-commerce) dispute/chargeback process — the fragment covers only ATM... |
| `BOOK_APPOINTMENT` | I want to book a branch appointment about a loan, how do I do that? | Booking branch appointments for non-investment topics |
| `CALL_BRANCH_AUTHORISED` | What is the phone number for Slovenská sporiteľňa's client centre? | The actual Klientske centrum phone number(s) |
| `CALL_PHONE_AUTHORISED_V2` | What does it mean that a call via George is 'verified'? | Definition and mechanism of the 'verified/authenticated' call (the central topic implie... |
| `CARD_CREDIT@EXTRAS` | How do I turn off automatic repayment of the full amount owed from my account? | deactivation/modification of the automatic full-repayment service |
| `CARD_CREDIT@REPAY` | What interest rate do I have on my credit card if I don't repay the full amount? | credit-card interest rate / APR |
| `CARD_CREDIT@APPLICATION` | What income or what conditions do I need to meet to get a credit card approved? | concrete eligibility conditions (age range, income/creditworthiness) |
| `CARD_CREDIT@APPLICATION` | How long does it take to decide on my credit-card application? | decision/approval timeframe |
| `CARD_CREDIT@DRAWDOWN` | How much does a cash withdrawal from an ATM with the credit card cost me? | ATM cash-withdrawal fee for credit cards |
| `CARD_CREDIT@ABOUT` | How many credit cards can I have at once at Slovenská sporiteľňa? | maximum number of credit cards per client |
| `CARD_CREDIT@APPLICATION_STATUS` | Where can I check the status of my credit-card application before the letter arrives? | proactive status-check channel (George / Klientske centrum) and decision timeframe |
| `CARD_DEBIT@SUBSCRIPTION_CANCEL` | How do I block payments to a specific merchant, e.g. Netflix, so it can no longer charg... | Blocking card transactions of a specific merchant (promised in summary, absent from des... |
| `CARD_DEBIT@CAPABILITIES` | How many digits does the ePIN I set in George have? | Consistent ePIN length (fragment says both four and five digits) |
| `CARD_DEBIT@ABOUT` | Do you charge any fees or surcharges when I pay by card abroad or in a foreign currency? | Foreign-transaction fees and currency-conversion terms |
| `CARD_DEBIT@CARD_CUSTOMIZATION` | Where do I upload my own photo for the card and how do I find out if it was approved? | image submission/upload channel and approval notification |
| `CARD_DEBIT@FEES` | What is the fee for paying with the card in a foreign currency or abroad? | foreign-currency / cross-border transaction fee amount |
| `CARD_DEBIT@FEES` | Do I pay any monthly fee for the debit card? | monthly debit card fee amount |
| `CHANGE_CARD_LIMITS` | What is the maximum daily ATM withdrawal limit on my card? | maximum daily ATM withdrawal limit in EUR |
| `CHANGE_DELIVERY_ADDRESS` | My card is already being produced — can I still change the delivery address, and by when? | Cut-off / deadline for changing the delivery address of an in-production card |
| `CLOSE_ACCOUNT` | When I close my savings account, do I also get the interest that has accrued? | Treatment and payout of accrued interest at the moment of closing the savings account. |
| `CREATE_SWEEP_ORDER` | How much does it cost to set up an above-limit standing order? | concrete setup fee amount for an above-limit standing order |
| `DATEIO@CLAIMS` | By what deadline must I file a Moneyback claim if the reward didn't arrive? | Claim filing deadline / time window |
| `EASY_ACCESS` | Where do I get the activation code when I no longer have my old phone? | Source of the activation code when no other active device exists (branch / Client Centre). |
| `FRAUD@ABOUT` | Someone took money from my account — will I get it back? | The process and conditions for recovering / refunding fraudulently taken funds. |
| `FRAUD@ABOUT` | How do I block my entire online banking and access to George? | Steps to block the account / online banking / George access (the description only cover... |
| `FX_CONVERTER` | How much does the bank charge for buying foreign cash at the branch, what is the fee? | fee/margin for buying foreign currency cash (valuty) at the branch |
| `GEORGE@WHAT_YOU_DO` | Does George save the history of my conversations with it? | Whether chat/conversation history is stored |
| `GEORGE@DAILY_BALANCE` | How much does one SMS notification about an account movement cost me? | Exact per-SMS notification price (only a tariff PDF link is given) |
| `GEORGE@DAILY_BALANCE` | How do I set up payment notifications in the George mobile app? | Mobile-app notification setup (this fragment is explicitly web-only) |
| `GEORGE_DEVICES` | How do I remove an old or lost device from my list of logged-in devices? | Removing/deactivating a device from the list |
| `GEORGE_INVEST` | What is the difference between the Standard, Trade and Explore programs? | Description/comparison of the three portfolio programs (only their names are listed). |
| `GIRO@CHILDREN_APPLICATION` | Can I open an account for a child younger than 6 years old? | Whether and how an account can be opened for a child under 6 years |
| `GIRO@STANDARD_PAYMENT_CARD` | How much does a second payment card to the SPACE account cost? | The actual fee for a second/additional card (only a link to the fee schedule is given) |
| `GIRO@STANDARD_ABOUT` | How much does it cost to maintain the SPACE account per month? | Monthly account maintenance fee / conditions for it to be free |
| `GIRO@STANDARD_APPLICATION` | What do I need to do before closing the SPACE account — withdraw money or cancel cards? | Prerequisites for account closure (balance, linked cards/products, overdraft) |
| `GIRO@STANDARD_DISPOSING_PERSON` | What can an authorised person do on my account — can they send payments and withdraw mo... | The specific scope of operations permitted to a disponent |
| `GIRO@FOREIGN_APPLICATION` | In which currencies can I open an account with you? | List of available foreign-currency account currencies |
| `GIRO@FOREIGN_FEES` | Is there a fee for opening a foreign-currency account? | account-opening fee for the foreign-currency account |
| `GIRO@FOREIGN_FEES` | If a payment in dollars comes into the account, do I pay any fee for that? | incoming/outgoing foreign-currency transfer and conversion fees |
| `GIRO@FOREIGN_PAYMENT_CARD` | How much does the card for a dollar account cost per month? | monthly card fee amount |

*(+80 more HIGH-priority gap questions in the CSV.)*

---
## 3. Fully-answerable additions (extra coverage)
64 proposed questions the KB *can* answer — natural high-probability phrasings not yet tested. Good regression coverage once gold answers are written. Examples:

- **`SAVING@FEES`** — Do I pay any fee when sending money from savings to another bank?
- **`SAVING@DISPOSING_PERSON`** — Can my underage child be a disponent on the savings account?
- **`SAVING@INTEREST_RATES_AND_LIMITS`** — Does the better rate count if I invest in ETFs, or only mutual funds?
- **`SAVING@ABOUT`** — Can I open a savings account in a currency other than euro?
- **`SAVING@ABOUT`** — Is the savings account protected if the bank fails?
- **`SAVING@PRODUCT_INFO`** — Can I rename my savings or change its colour in George?
- **`SAVING@KIDS_APPLICATION`** — Can I cancel the Piggy Bank via George too, or only at a branch?
- **`SAVING@KIDS_ABOUT`** — Does the child need the George Junior app for the Piggy Bank, or is a SPACE account enough?
- **`SAVING@ACCOUNT_STATEMENTS`** — How far back can I download savings statements?
- **`SAVING@DEPOSIT_INTEREST_RATES`** — Can my term deposit interest rate change during the fixed term?
- **`SAVING@DEPOSIT_ACCOUNT_MOVEMENTS`** — How much does it cost when another person deposits cash to my deposit account?
- **`SAVING@DEPOSIT_ACCOUNT_STATEMENTS`** — Why do I get a term-deposit statement on every movement?

See `new_test_cases.csv` (`comment` contains `[answerable]`) for all of them.
