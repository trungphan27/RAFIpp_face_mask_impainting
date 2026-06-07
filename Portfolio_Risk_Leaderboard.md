## 1. Sequence Diagram
```mermaid
sequenceDiagram
    actor User as Trader/User
    participant UI as Portfolio.jsx
    participant PS as portfolioStore.js
    participant SC as Supabase Client
    participant DB as Supabase Tables
    participant LS as leaderboardStore.js

    User->>UI: Buy/Sell stock
    UI->>PS: buy() / sell()
    PS->>PS: Update cash, holdings, transactions
    PS->>SC: get authenticated user
    SC->>DB: Upsert portfolios
    SC->>DB: Insert transactions
    PS->>LS: submitScore(portfolioValue, totalReturn, tradeCount)
    LS->>SC: get user profile
    SC->>DB: Select user_profiles
    LS->>SC: upsert leaderboard score
    SC->>DB: Upsert leaderboard_entries
    LS->>SC: fetch leaderboard
    SC->>DB: Select leaderboard_entries
    LS-->>UI: Updated entries and user rank
```

# Portfolio, Risk, and Leaderboard Component - UML Class Diagram

## Relationship Legend

| UML relationship | Mermaid syntax | Meaning |
|---|---:|---|
| Association | `A --> B` | A has a stable structural/use relationship with B. |
| Dependency | `A ..> B` | A temporarily calls or depends on B's API. |
| Aggregation | `A o-- B` | A groups B, but B can exist independently. |
| Composition | `A *-- B` | A strongly owns B as part of its state/lifecycle. |
| Inheritance | `Parent <|-- Child` | Child extends Parent. Not used in the current implementation. |
| Realization / Implementation | `Interface <|.. Class` | Class implements Interface. Not used in the current implementation. |

## Class Diagram

```mermaid
classDiagram
    direction LR

    class PortfolioPage {
        <<React Page>>
        +render()
        +downloadCSV()
    }

    class LeaderboardPage {
        <<React Page>>
        +render()
        +changePeriod(period)
    }

    class PortfolioStore {
        <<Zustand Store>>
        -cash
        -initialCash
        -holdings
        -transactions
        -portfolioHistory
        -dailyPLHistory
        +buy(ticker, quantity, price, orderType)
        +sell(ticker, quantity, price, orderType)
        +syncToSupabase(tx)
        +loadFromSupabase()
        +getPortfolioValue(prices)
        +getUnrealizedPL(prices)
        +getTotalRealizedPL()
        +getTotalReturn(prices)
        +getHoldingsArray(prices)
        +getAllocation(prices)
        +recordSnapshot(prices)
        +submitToLeaderboard(prices)
        +reset()
    }

    class LeaderboardStore {
        <<Zustand Store>>
        -entries
        -period
        -userRank
        -loaded
        +setPeriod(period)
        +setEntries(entries)
        +fetchFromSupabase()
        +submitScore(portfolioValue, totalReturn, tradesCount)
    }

    class MarketStore {
        <<Zustand Store>>
        -prices
        -prevPrices
        -rawTicks
        +getChange(ticker)
        +getOHLCV(ticker, timeframe)
        +updatePrices(data)
        +simulateTick()
    }

    class AuthStore {
        <<Zustand Store>>
        -user
        -session
    }

    class SupabaseClient {
        <<External Client>>
        +auth.getUser()
        +from(table)
        +isSupabaseConfigured()
    }

    class PortfolioRoute {
        <<Express Router>>
        +GET /api/portfolio
        +GET /api/portfolio/history
        +GET /api/portfolio/risk
    }

    class LeaderboardRoute {
        <<Express Router>>
        +GET /api/leaderboard
    }

    class RiskMetrics {
        <<Service>>
        +sharpeRatio(returns, riskFreeRate)
        +maxDrawdown(values)
        +volatility(returns, annualized)
        +beta(portfolioReturns, marketReturns)
        +winRate(trades)
        +profitFactor(trades)
    }

    class SimulationEngine {
        <<Service>>
        -stocks
        -prices
        -rawTicks
        -regime
        +tick()
        +start(intervalMs)
        +stop()
        +onTick(callback)
        +setRegime(regime, params)
        +applyShock(ticker, shockPercent)
        +getQuote(ticker)
    }

    class Portfolio {
        <<Data Model>>
        +userId
        +cash
        +initialCash
        +holdings
        +updatedAt
    }

    class Holding {
        <<Data Model>>
        +ticker
        +shares
        +avgPrice
        +realizedPL
    }

    class Transaction {
        <<Data Model>>
        +id
        +type
        +ticker
        +orderType
        +quantity
        +price
        +total
        +time
        +status
    }

    class LeaderboardEntry {
        <<Data Model>>
        +userId
        +displayName
        +portfolioValue
        +totalReturn
        +sharpeRatio
        +tradesCount
        +period
        +updatedAt
    }

    class UserProfile {
        <<Data Model>>
        +id
        +displayName
        +watchlist
    }

    PortfolioPage --> PortfolioStore : association - reads portfolio state
    PortfolioPage --> MarketStore : association - reads live prices
    LeaderboardPage --> LeaderboardStore : association - reads ranking state
    LeaderboardPage --> AuthStore : association - checks current user

    PortfolioStore *-- Portfolio : composition - owns portfolio state
    Portfolio *-- Holding : composition - contains holdings
    Portfolio *-- Transaction : composition - contains transactions
    PortfolioStore ..> MarketStore : dependency - calculates with prices
    PortfolioStore ..> SupabaseClient : dependency - sync/load portfolio
    PortfolioStore ..> LeaderboardStore : dependency - submits score

    LeaderboardStore o-- LeaderboardEntry : aggregation - groups fetched entries
    LeaderboardStore ..> SupabaseClient : dependency - fetch/upsert scores
    AuthStore ..> SupabaseClient : dependency - auth session

    SupabaseClient --> Portfolio : association - portfolios table
    SupabaseClient --> Transaction : association - transactions table
    SupabaseClient --> LeaderboardEntry : association - leaderboard_entries table
    SupabaseClient --> UserProfile : association - user_profiles table

    LeaderboardEntry --> UserProfile : association - belongs to user profile

    PortfolioRoute o-- Portfolio : aggregation - reads in-memory portfolio
    PortfolioRoute ..> SimulationEngine : dependency - current holding values
    PortfolioRoute ..> RiskMetrics : dependency - risk calculations
    LeaderboardRoute o-- LeaderboardEntry : aggregation - returns ranking rows
```

## Notes

- The current source code does not define inheritance or interface implementation for this component.
- `Composition` is used where the object is part of the owning state, such as `Portfolio` containing `Holding` and `Transaction`.
- `Aggregation` is used where a class groups data that can exist independently, such as `LeaderboardStore` grouping `LeaderboardEntry`.
- `Dependency` is used for API/service calls, such as `PortfolioStore` calling `SupabaseClient` or `RiskMetrics`.