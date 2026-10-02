from big_two_core import BigTwoGame, Card, get_deck, evaluate_hand, compare_hands, RANKS, SUITS

def test_card_values():
    """測試牌的數值和排序是否符合 Big Two 規則 (3D < ... < 2S)。"""
    c3D = Card('3', 'D')
    c3S = Card('3', 'S')
    c2D = Card('2', 'D')
    c2S = Card('2', 'S')
    
    assert c3D < c3S
    assert c3S < c2D
    assert c2D < c2S
    assert c2S > c3D
    
    print("Card value and comparison test passed.")

def test_evaluate_hand():
    """測試 evaluate_hand 函式是否能正確識別牌型。"""
    # 單張
    single = [Card('A', 'S')]
    assert evaluate_hand(single)[0] == 'single'
    
    # 對子
    pair = [Card('3', 'D'), Card('3', 'C')]
    assert evaluate_hand(pair)[0] == 'pair'
    
    # 三條
    triple = [Card('5', 'D'), Card('5', 'C'), Card('5', 'H')]
    assert evaluate_hand(triple)[0] == 'triple'
    
    # 順子 (3-4-5-6-7)
    straight = [Card('3', 'D'), Card('4', 'C'), Card('5', 'H'), Card('6', 'S'), Card('7', 'D')]
    assert evaluate_hand(straight)[0] == 'straight'
    
    # 順子 (J-Q-K-A-2) - 最大的順子
    straight_max = [Card('J', 'D'), Card('Q', 'C'), Card('K', 'H'), Card('A', 'S'), Card('2', 'D')]
    assert evaluate_hand(straight_max)[0] == 'straight'
    
    # 同花 (梅花)
    flush = [Card('3', 'C'), Card('5', 'C'), Card('8', 'C'), Card('J', 'C'), Card('A', 'C')]
    assert evaluate_hand(flush)[0] == 'flush'
    
    # 葫蘆 (A-A-A-K-K)
    full_house = [Card('A', 'D'), Card('A', 'C'), Card('A', 'H'), Card('K', 'S'), Card('K', 'D')]
    assert evaluate_hand(full_house)[0] == 'full_house'
    
    # 鐵支 (2-2-2-2-3)
    four_of_a_kind = [Card('2', 'D'), Card('2', 'C'), Card('2', 'H'), Card('2', 'S'), Card('3', 'D')]
    assert evaluate_hand(four_of_a_kind)[0] == 'four_of_a_kind'
    
    # 同花順 (3-4-5-6-7 of Diamonds)
    straight_flush = [Card('3', 'D'), Card('4', 'D'), Card('5', 'D'), Card('6', 'D'), Card('7', 'D')]
    assert evaluate_hand(straight_flush)[0] == 'straight_flush'
    
    # 無效牌型 (四張)
    invalid = [Card('3', 'D'), Card('4', 'C'), Card('5', 'H'), Card('6', 'S')]
    assert evaluate_hand(invalid) is None
    
    print("Hand evaluation test passed.")

def test_compare_hands():
    """測試 compare_hands 函式是否能正確比較牌型大小。"""
    # 單張比較
    c3D = [Card('3', 'D')]
    c3S = [Card('3', 'S')]
    c4D = [Card('4', 'D')]
    
    assert compare_hands(c3S, c3D) == 1 # 3S > 3D
    assert compare_hands(c4D, c3S) == 1 # 4D > 3S
    
    # 牌型比較 (順子 > 三條)
    straight = [Card('3', 'D'), Card('4', 'C'), Card('5', 'H'), Card('6', 'S'), Card('7', 'D')]
    triple = [Card('5', 'D'), Card('5', 'C'), Card('5', 'H'), Card('3', 'D'), Card('4', 'C')] # 無效牌型，應為 None
    
    # 修正：使用有效的五張牌型
    straight = [Card('3', 'D'), Card('4', 'C'), Card('5', 'H'), Card('6', 'S'), Card('7', 'D')]
    full_house = [Card('A', 'D'), Card('A', 'C'), Card('A', 'H'), Card('K', 'S'), Card('K', 'D')]
    
    assert compare_hands(straight, full_house) == -1 # 順子 < 葫蘆
    
    # 同牌型比較 (葫蘆 vs 葫蘆)
    fh1 = [Card('A', 'D'), Card('A', 'C'), Card('A', 'H'), Card('K', 'S'), Card('K', 'D')] # AAAKK
    fh2 = [Card('K', 'D'), Card('K', 'C'), Card('K', 'H'), Card('Q', 'S'), Card('Q', 'D')] # KKKQQ
    
    assert compare_hands(fh1, fh2) == 1 # AAAKK > KKKQQ
    
    print("Hand comparison test passed.")

def test_game_flow_and_legal_moves():
    """測試遊戲流程和合法動作產生器。"""
    game = BigTwoGame()
    game.reset()
    
    # 找到持有梅花 3 的玩家
    start_player = game.current_player
    hand = game.hands[start_player]
    
    # 測試起始玩家的合法動作 (必須包含梅花 3)
    legal_moves = game.get_legal_moves(hand, game.last_played_hand)
    
    # 確保 Pass 不在起始動作中
    assert [] not in legal_moves
    
    # 確保所有合法動作都包含梅花 3
    for move in legal_moves:
        assert Card('3', 'C') in move
        
    # 執行一個動作 (單張梅花 3)
    action = [Card('3', 'C')]
    new_state, reward, done, info = game.step(action)
    
    assert game.last_played_hand == action
    assert game.last_played_player == start_player
    assert game.current_player == (start_player + 1) % 4
    
    # 測試下一位玩家的合法動作 (必須大於梅花 3，且是單張)
    next_player = game.current_player
    next_hand = game.hands[next_player]
    legal_moves_next = game.get_legal_moves(next_hand, game.last_played_hand)
    
    # 確保 Pass 在合法動作中
    assert [] in legal_moves_next
    
    # 確保所有非 Pass 動作都是單張且大於梅花 3
    for move in legal_moves_next:
        if move:
            assert len(move) == 1
            assert compare_hands(move, action) == 1
            
    print("Game flow and legal moves test passed.")

if __name__ == '__main__':
    test_card_values()
    test_evaluate_hand()
    test_compare_hands()
    test_game_flow_and_legal_moves()
