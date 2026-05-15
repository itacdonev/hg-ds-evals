- 32/513 bot responses drift into Czech — a real bug, not a valid Slovak variant.

**KEY FAILURE MODES VISIBLE IN THE AGENT PROMPTS BELOW**
> - `main_agent` lists languages as "Czech, English, German, Slovak" — Czech
>   first. For empty queries or ambiguous language, the default is Czech.
> - `daily_banking_agent` identifies as *Česká spořitelna* (Czech bank), not
>   *Slovenská sporiteľňa*. Czech pronoun examples are given ("Vy in Czech,
>   Sie in German"); no Slovak equivalent.
> - Links point at `george.csas.cz` (Czech domain).