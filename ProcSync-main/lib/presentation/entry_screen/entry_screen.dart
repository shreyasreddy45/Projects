import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import '../../core/app_export.dart';
import '../chat/models/user_model.dart';
import '../dashboard_page/models/class_model.dart';

class EntryScreen extends StatefulWidget {
  const EntryScreen({super.key});

  @override
  State<EntryScreen> createState() => _EntryScreenState();
}

class _EntryScreenState extends State<EntryScreen> {
  @override
  void initState() {
    super.initState();
    _checkAuthAndNavigate();
  }

  Future<void> _checkAuthAndNavigate() async {
    try {
      final user = FirebaseAuth.instance.currentUser;

      if (user == null) {
        Navigator.pushReplacementNamed(context, AppRoutes.loginScreen);
        return;
      }

      final userDoc = await FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .get();

      if (!userDoc.exists) {
        Navigator.pushReplacementNamed(context, AppRoutes.accScreen);
        return;
      }

      final data = userDoc.data()!;
      final role = data['role'] as String?;
      final joinedClassId = data['joinedClassId'] as String?;

      final userModel = UserModel(
        uid: data['uid'],
        name: data['name'],
        email: data['email'],
        role: role ?? '',
        joinedClassId: joinedClassId,
      );

      if (role == 'proctor') {
        Navigator.pushReplacementNamed(context, AppRoutes.dashboardPage);
      } else if (role == 'student') {
        if (joinedClassId == null || joinedClassId.isEmpty) {
          Navigator.pushReplacementNamed(context, AppRoutes.studentJoinPage);
        } else {
          final classDoc = await FirebaseFirestore.instance
              .collection('classes')
              .doc(joinedClassId)
              .get();

          if (classDoc.exists) {
            final classModel =
                ClassModel.fromMap(classDoc.data()!, id: classDoc.id);
            Navigator.pushReplacementNamed(context, AppRoutes.chatPage,
                arguments: {
                  'user': userModel,
                  'classModel': classModel,
                });
          } else {
            Navigator.pushReplacementNamed(context, AppRoutes.studentJoinPage);
          }
        }
      } else {
        Navigator.pushReplacementNamed(context, AppRoutes.accScreen);
      }
    } catch (e) {
      // On any error fallback to login
      Navigator.pushReplacementNamed(context, AppRoutes.loginScreen);
    }
  }

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(child: CircularProgressIndicator()),
    );
  }
}
